import argparse
import csv
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


def aggregate_lm_evaluation_harness_results(
    results_root_dir: str, model_names: list[str]
) -> dict[str, set[str]]:
    model_wrong_qids = {}

    for model_name in tqdm(model_names):
        wrong_qids = set()

        result_file = list(Path(results_root_dir, model_name).glob("*.jsonl"))[0]
        with open(result_file) as f:
            pred_items = json.load(f)
            for pred_item in pred_items:
                if pred_item["acc"] == 0.0:
                    wrong_qids.add(pred_item["doc"]["qid"])

        model_wrong_qids[model_name] = wrong_qids

    return model_wrong_qids


def aggregate_llm_jp_eval_results(
    results_root_dir: str, model_names: list[str], dataset: Dataset
) -> dict[str, set[str]]:
    model_wrong_qids = {}

    for model_name in tqdm(model_names):
        wrong_qids = set()

        result_file = list(Path(results_root_dir, model_name).iterdir())[0]
        with open(result_file) as f:
            result = json.load(f)

            task_names = list(result["outputs"].keys())
            assert len(task_names) == 1

            pred_items = result["outputs"][task_names[0]]
            assert len(pred_items) == len(dataset)

            for i in range(len(dataset)):
                example = dataset[i]
                pred_item = pred_items[i]
                assert example["question"] in pred_item["input"]

                if pred_item["pred"] != pred_item["gold"]:
                    wrong_qids.add(example["qid"])

        model_wrong_qids[model_name] = wrong_qids

    return model_wrong_qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config_name", type=str)
    parser.add_argument("--dataset_split_name", type=str)
    parser.add_argument("--type", choices=("lm-evaluation-harness", "llm-jp-eval"), required=True)
    parser.add_argument("--results_root_dir", type=str, required=True)
    parser.add_argument("--model_names", type=str, nargs="+", required=True)
    parser.add_argument("--invalid_qids_file", type=str)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    model_names = args.model_names

    dataset = load_dataset(args.dataset_name, name=args.dataset_config_name, split=args.dataset_split_name)
    assert not isinstance(dataset, DatasetDict)

    invalid_qids = set()
    if args.invalid_qids_file is not None:
        with open(args.invalid_qids_file) as f:
            for line in f:
                invalid_qids.add(line.strip())

    if args.type == "lm-evaluation-harness":
        model_wrong_qids = aggregate_lm_evaluation_harness_results(args.results_root_dir, model_names)
    elif args.type == "llm-jp-eval":
        model_wrong_qids = aggregate_llm_jp_eval_results(args.results_root_dir, model_names, dataset)

    num_examples = len(dataset)
    num_examples_filtered = 0
    num_corrects = {model_name: 0 for model_name in model_names}
    num_corrects_filtered = {model_name: 0 for model_name in model_names}

    with open(args.output_file, "w", newline="") as fo:
        fieldnames = dataset.column_names + ["is_valid"] + model_names
        writer = csv.DictWriter(fo, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer.writeheader()

        for i in range(num_examples):
            example = dict(**dataset[i])
            qid = example["qid"]
            is_valid = qid not in invalid_qids
            example["is_valid"] = int(is_valid)
            if is_valid:
                num_examples_filtered += 1

            for model_name in model_names:
                is_correct = qid not in model_wrong_qids[model_name]
                example[model_name] = int(is_correct)
                if is_correct:
                    num_corrects[model_name] += 1
                    if is_valid:
                        num_corrects_filtered[model_name] += 1

            writer.writerow(example)

    print("model_name\tacc\tacc_filtered")
    for model_name in model_names:
        acc = num_corrects[model_name] / num_examples
        acc_filtered = num_corrects_filtered[model_name] / num_examples_filtered
        print(model_name, f"{acc:.3f}", f"{acc_filtered:.3f}", sep="\t")

if __name__ == "__main__":
    main()
