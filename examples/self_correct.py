"""
Zero-config self-correction with the shipped base probe.

Requirements:
    pip install "sottovoce[train]"   # transformers is needed to load the model

This script:
1. Loads Qwen 2.5 3B Instruct.
2. Downloads and loads the pre-trained base probe (one line, no training).
3. Runs two-pass self-correction on a few questions and shows which answers
   the probe flagged and the model revised.

For training your own probe from scratch, see quickstart.py.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from sottovoce import SelfCorrector, load_base_probe


def main():
    model_id = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Download + load the pre-trained base probe (cached after first use).
    probe = load_base_probe("qwen2.5-3b")
    corrector = SelfCorrector(model, tokenizer, probe)

    questions = [
        "What is the capital of France?",           # confident, correct
        "Who invented the transistor?",             # answerable
        "What is the airspeed velocity of an unladen swallow?",  # unanswerable
    ]

    for q in questions:
        result = corrector.generate(q)
        print(f"\nQ: {q}")
        print(f"A: {result.response}")
        print(f"   probe score {result.probe_score:.3f} -> {result.decision.value}")
        if result.was_corrected:
            print(f"   [revised from: {result.original_response[:60]}...]")


if __name__ == "__main__":
    main()
