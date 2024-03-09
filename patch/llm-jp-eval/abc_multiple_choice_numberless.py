import csv
from pathlib import Path

from huggingface_hub import hf_hub_download

from llm_jp_eval.datasets.base import BaseDatasetProcessor, Sample


class AbcMultipleChoiceNumberlessDatasetProcessor(BaseDatasetProcessor):
    data_name = "abc_multiple_choice_numberless"

    def __init__(self, dataset_dir: Path, version_name: str) -> None:
        super().__init__(dataset_dir, version_name)
        self.output_info.instruction = (
            "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。回答の他には何も含めないことを厳守してください。"
        )
        self.output_info.output_length = 30
        self.output_info.metrics = ["exact_match"]

    def download(self):
        hf_dataset_name = "tohoku-nlp/abc-multiple-choice"
        hf_train_filename = "train.tsv"
        hf_test_filename = "test.tsv"

        raw_train_path: Path = self.raw_dir / f"{self.data_name}_train.tsv"
        if not raw_train_path.exists():
            hf_hub_download(hf_dataset_name, hf_train_filename, repo_type="dataset", local_dir=self.raw_dir)
            (self.raw_dir / hf_train_filename).rename(raw_train_path)

        raw_test_path: Path = self.raw_dir / f"{self.data_name}_test.tsv"
        if not raw_test_path.exists():
            hf_hub_download(hf_dataset_name, hf_test_filename, repo_type="dataset", local_dir=self.raw_dir)
            (self.raw_dir / hf_test_filename).rename(raw_test_path)

    def preprocess_evaluation_data(self):
        train_samples: list[Sample] = []
        with (self.raw_dir / f"{self.data_name}_train.tsv").open(encoding="utf-8") as f_train:
            reader = csv.DictReader(f_train, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True)
            for row in reader:
                train_samples.append(
                    Sample(
                        input=f"質問：{row['question']}\n選択肢：{row['choice_1']},{row['choice_2']},{row['choice_3']},{row['choice_4']}",
                        output=row[f"choice_{row['answer']}"],
                    )
                )
        self._save_evaluation_data(train_samples, self.evaluation_dir / "train" / f"{self.data_name}.json")

        test_samples: list[Sample] = []
        with (self.raw_dir / f"{self.data_name}_test.tsv").open(encoding="utf-8") as f_test:
            reader = csv.DictReader(f_test, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True)
            for row in reader:
                test_samples.append(
                    Sample(
                        input=f"質問：{row['question']}\n選択肢：{row['choice_1']},{row['choice_2']},{row['choice_3']},{row['choice_4']}",
                        output=row[f"choice_{row['answer']}"],
                    )
                )
        self._save_evaluation_data(test_samples, self.evaluation_dir / "test" / f"{self.data_name}.json")
