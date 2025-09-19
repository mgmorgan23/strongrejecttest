import argparse
import os
import pathlib

from datasets import concatenate_datasets
from dotenv import load_dotenv

from strong_reject.load_datasets import load_strongreject
from strong_reject.evaluate import evaluate_dataset
from strong_reject.generate import generate_to_dataset
from strong_reject.jailbreaks import (
    apply_jailbreaks_to_dataset,
    decode_dataset,
    registered_jailbreaks,
)
jailbreaks_for_eval = [
    "rot_13",
    # "disemvowel",
    # "gcg_transfer_harmbench",
    # "gcg_transfer_universal_attacks",
    # "combination_3",
    # "combination_2",
    # "few_shot_json",
    # "dev_mode_v2",
    # "dev_mode_with_rant",
    # "wikipedia_with_title",
    # "distractors",
    # "wikipedia",
    # "style_injection_json",
    # "style_injection_short",
    # "refusal_suppression",
    # "prefix_injection",
    # "distractors_negated",
    # "poems",
    # "base64",
    # "base64_raw",
    # "base64_input_only",
    # "base64_output_only",
    # "evil_confidant",
    # "none",
    # "aim"
]
registered_jailbreaks_for_eval = {}

for jail in registered_jailbreaks.keys():
    if jail in jailbreaks_for_eval:
        registered_jailbreaks_for_eval[jail] = registered_jailbreaks[jail]
DATA_DIR = "/results"

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")


def main():
    if os.getenv("TESTING"):
        dataset = load_strongreject().shuffle().select(range(2))
        jailbreaks = "base64", "translation_hmong", "pair"
        models = (
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        )
        max_tokens = 100
    else:
        dataset = load_strongreject()
        jailbreaks = registered_jailbreaks_for_eval
        models = (
            "Qwen/Qwen2-7B-Instruct
        )
        max_tokens = 512

    dataset = apply_jailbreaks_to_dataset(dataset, jailbreaks)
    dataset.to_json(str(DATA_DIR / "interim" / "jailbreaks.json"))
    datasets = []
    for model in models:
        model_dataset = generate_to_dataset(
            dataset, models=[model], target_column="jailbroken_prompt", max_tokens=max_tokens
        )
        model_dataset.to_json(str(DATA_DIR / "interim" / f"responses_{model}.json"))
        model_dataset = evaluate_dataset(model_dataset, ["strongreject_finetuned"])
        model_dataset.to_json(str(DATA_DIR / "interim" / f"evals_{model}.json"))
        datasets.append(model_dataset)

    return concatenate_datasets(datasets)


if __name__ == "__main__":
    load_dotenv()
    filename = "evals.json"
    args = parser.parse_args()
    if args.test:
        os.environ["TESTING"] = "true"
        filename = "evals_test.json"

    dataset = main()
    dataset.to_json(str(DATA_DIR / "interim" / filename))
