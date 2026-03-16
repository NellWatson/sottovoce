"""
Train a calibration probe on a language model's residual stream activations.

Usage:
    python -m sottovoce.train \
        --model Qwen/Qwen2.5-3B-Instruct \
        --dataset triviaqa \
        --n-samples 2000 \
        --output probes/my_probe.pt

Watson & Claude (2026). "The Model Already Knows: Universal Uncertainty
Signals in Language Model Residual Streams."
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sottovoce.probe import _ProbeNet

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    quantize: bool = False,
    device: str = "auto",
):
    """Load a causal LM and tokenizer, optionally quantized to 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": device, "trust_remote_code": True}

    if quantize:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["low_cpu_mem_usage"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_triviaqa(n_samples: int, seed: int = 42) -> list[dict]:
    """Load TriviaQA questions with answers."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc", split="validation")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    samples = []
    for idx in indices:
        item = ds[int(idx)]
        question = item["question"]
        aliases = item["answer"]["aliases"]
        normalized = item["answer"].get("normalized_aliases", aliases)
        samples.append({
            "question": question,
            "aliases": aliases,
            "normalized_aliases": normalized,
        })
    return samples


def generate_and_judge(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 64,
) -> tuple[list[str], list[bool]]:
    """Generate answers and judge correctness against aliases."""
    prompts = []
    for s in samples:
        prompts.append(
            f"Answer the following question in one sentence.\n\n"
            f"Question: {s['question']}\nAnswer:"
        )

    responses = []
    correct = []

    device = next(model.parameters()).device

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

        response_lower = response.lower()
        aliases = samples[i].get("normalized_aliases", samples[i]["aliases"])
        is_correct = any(a.lower() in response_lower for a in aliases)
        correct.append(is_correct)

        if (i + 1) % 100 == 0:
            acc = sum(correct) / len(correct)
            logger.info(f"Generated {i+1}/{len(samples)}, running accuracy: {acc:.3f}")

    return responses, correct


def extract_features_batch(
    model,
    tokenizer,
    texts: list[str],
    probe_layer: int,
) -> np.ndarray:
    """Extract residual stream features at a given layer for all texts."""
    features = []
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["act"] = hidden[0, -1, :].detach().cpu().float()

    handle = model.model.layers[probe_layer].register_forward_hook(hook_fn)
    device = next(model.parameters()).device

    try:
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs, use_cache=False)

            if "act" in captured:
                features.append(captured["act"].numpy())
                del captured["act"]

            if (i + 1) % 100 == 0:
                logger.info(f"Extracted features: {i+1}/{len(texts)}")
    finally:
        handle.remove()

    return np.stack(features)


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    epochs: int = 20,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[_ProbeNet, dict]:
    """Train a 2-layer MLP probe on extracted features."""
    rng = np.random.default_rng(seed)
    n = len(features)
    indices = rng.permutation(n)
    val_n = int(n * val_split)

    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    X_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.float32).unsqueeze(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = features.shape[1]
    probe = _ProbeNet(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    probe.to(device)
    probe.train()

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    metrics = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = probe(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

        from sklearn.metrics import roc_auc_score
        try:
            val_auroc = roc_auc_score(labels[val_idx], val_probs)
        except ValueError:
            val_auroc = 0.5

        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_auroc"].append(val_auroc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_auroc={val_auroc:.4f}"
            )

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    probe.cpu()

    return probe, metrics


def compute_final_metrics(
    probe: _ProbeNet,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute AUROC, ECE, accuracy on full dataset."""
    from sklearn.metrics import roc_auc_score

    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(probe(X)).numpy().flatten()

    auroc = roc_auc_score(labels, probs)

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    predictions = (probs >= 0.5).astype(float)
    accuracy = (predictions == labels).mean()

    return {
        "auroc": float(auroc),
        "ece": float(ece),
        "accuracy": float(accuracy),
        "n_samples": int(len(labels)),
        "n_correct_answers": int(labels.sum()),
        "base_rate": float(labels.mean()),
    }


def main(args: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Train a calibration probe on a language model's residual stream.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", default="triviaqa", choices=["triviaqa"],
                        help="Dataset to use for training")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--output", required=True, help="Path to save probe .pt file")
    parser.add_argument("--layer-fraction", type=float, default=0.67,
                        help="Fraction of model depth to probe")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Probe MLP hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split fraction")
    parser.add_argument("--quantize", action="store_true",
                        help="Load model in 4-bit (for large models)")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Max tokens per generated answer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-features", help="Save extracted features to .npz")
    parser.add_argument("--save-metrics", help="Save training metrics to .json")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    parsed = parser.parse_args(args)

    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info(f"Loading model: {parsed.model}")
    model, tokenizer = load_model_and_tokenizer(
        parsed.model, quantize=parsed.quantize,
    )

    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    probe_layer = int(n_layers * parsed.layer_fraction)
    logger.info(
        f"Model: {n_layers} layers, hidden_size={hidden_dim}, "
        f"probing layer {probe_layer}"
    )

    logger.info(f"Loading {parsed.dataset} with {parsed.n_samples} samples")
    if parsed.dataset == "triviaqa":
        samples = load_triviaqa(parsed.n_samples, seed=parsed.seed)
    else:
        raise ValueError(f"Unknown dataset: {parsed.dataset}")

    logger.info("Generating answers and judging correctness...")
    prompts = []
    for s in samples:
        prompts.append(
            f"Answer the following question in one sentence.\n\n"
            f"Question: {s['question']}\nAnswer:"
        )

    responses, correct = generate_and_judge(
        model, tokenizer, samples, max_new_tokens=parsed.max_new_tokens,
    )

    labels = np.array(correct, dtype=np.float32)
    logger.info(
        f"Generated {len(responses)} responses, "
        f"accuracy: {labels.mean():.3f} ({int(labels.sum())}/{len(labels)})"
    )

    full_texts = [
        f"{prompt} {response}"
        for prompt, response in zip(prompts, responses)
    ]

    logger.info(f"Extracting residual stream features at layer {probe_layer}...")
    features = extract_features_batch(model, tokenizer, full_texts, probe_layer)
    logger.info(f"Features shape: {features.shape}")

    if parsed.save_features:
        np.savez(parsed.save_features, features=features, labels=labels)
        logger.info(f"Saved features to {parsed.save_features}")

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Training probe...")
    probe_net, train_metrics = train_probe(
        features, labels,
        hidden_dim=parsed.hidden_dim,
        dropout=parsed.dropout,
        epochs=parsed.epochs,
        lr=parsed.lr,
        val_split=parsed.val_split,
        seed=parsed.seed,
    )

    final = compute_final_metrics(probe_net, features, labels)
    logger.info(
        f"Final: AUROC={final['auroc']:.4f}, "
        f"ECE={final['ece']:.4f}, "
        f"accuracy={final['accuracy']:.4f}"
    )

    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_net.state_dict(), str(output_path))
    logger.info(f"Saved probe to {output_path}")

    if parsed.save_metrics:
        metrics_out = {
            "model": parsed.model,
            "dataset": parsed.dataset,
            "n_samples": parsed.n_samples,
            "probe_layer": probe_layer,
            "layer_fraction": parsed.layer_fraction,
            "hidden_dim": parsed.hidden_dim,
            "final": final,
            "training": {
                "epochs": parsed.epochs,
                "best_val_auroc": max(train_metrics["val_auroc"]),
                "final_train_loss": train_metrics["train_loss"][-1],
                "final_val_loss": train_metrics["val_loss"][-1],
            },
        }
        with open(parsed.save_metrics, "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info(f"Saved metrics to {parsed.save_metrics}")

    print(f"\nProbe trained successfully.")
    print(f"  AUROC: {final['auroc']:.4f}")
    print(f"  ECE:   {final['ece']:.4f}")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
