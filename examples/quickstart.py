"""
Quickstart: Train a probe on Qwen 2.5 3B and test it.

Requirements:
    pip install sottovoce[train]

This script:
1. Loads Qwen 2.5 3B Instruct
2. Generates answers for 500 TriviaQA questions
3. Extracts residual stream features at layer 24 (~2/3 depth)
4. Trains a 2-layer MLP probe
5. Reports AUROC and ECE
6. Demonstrates routing decisions
"""

from sottovoce import CalibrationProbe
from sottovoce.train import (
    load_model_and_tokenizer,
    load_triviaqa,
    generate_and_judge,
    extract_features_batch,
    train_probe,
    compute_final_metrics,
)
import numpy as np
import torch


def main():
    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    N_SAMPLES = 500
    OUTPUT = "probes/qwen2.5-3b-demo.pt"

    # 1. Load model
    print(f"Loading {MODEL}...")
    model, tokenizer = load_model_and_tokenizer(MODEL)

    n_layers = len(model.model.layers)
    probe_layer = int(n_layers * 0.67)
    print(f"  {n_layers} layers, probing layer {probe_layer}")

    # 2. Generate and judge
    print(f"Generating answers for {N_SAMPLES} TriviaQA questions...")
    samples = load_triviaqa(N_SAMPLES)

    prompts = [
        f"Answer the following question in one sentence.\n\n"
        f"Question: {s['question']}\nAnswer:"
        for s in samples
    ]

    responses, correct = generate_and_judge(model, tokenizer, samples)
    labels = np.array(correct, dtype=np.float32)
    print(f"  Accuracy: {labels.mean():.3f}")

    # 3. Extract features
    print("Extracting residual stream features...")
    full_texts = [f"{p} {r}" for p, r in zip(prompts, responses)]
    features = extract_features_batch(model, tokenizer, full_texts, probe_layer)
    print(f"  Shape: {features.shape}")

    # 4. Train probe
    print("Training probe...")
    probe_net, metrics = train_probe(features, labels, epochs=20)

    # 5. Evaluate
    final = compute_final_metrics(probe_net, features, labels)
    print(f"\nResults:")
    print(f"  AUROC: {final['auroc']:.4f}")
    print(f"  ECE:   {final['ece']:.4f}")
    print(f"  Best val AUROC: {max(metrics['val_auroc']):.4f}")

    # 6. Save and reload as CalibrationProbe
    from pathlib import Path
    Path("probes").mkdir(exist_ok=True)
    torch.save(probe_net.state_dict(), OUTPUT)

    probe = CalibrationProbe.from_pretrained(OUTPUT)
    print(f"\nProbe loaded from {OUTPUT}")

    # 7. Demo: self-correction on a few questions
    from sottovoce import SelfCorrector

    corrector = SelfCorrector(model, tokenizer, probe)

    test_questions = [
        "What is the capital of France?",
        "Who invented the transistor?",
        "What is the airspeed velocity of an unladen swallow?",
    ]

    print("\nSelf-correction demo:")
    for q in test_questions:
        result = corrector.generate(q)
        print(f"\n  Q: {q}")
        print(f"  A: {result.response}")
        print(f"  Probe score: {result.probe_score:.3f} ({result.decision.value})")
        if result.was_corrected:
            print(f"  [Corrected from: {result.original_response[:60]}...]")


if __name__ == "__main__":
    main()
