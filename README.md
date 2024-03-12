# abc-multiple-choice Dataset

abc-multiple-choice は、競技クイズの大会「abc」で使用された4択問題を題材に作成された、多肢選択式の質問応答データセットです。

データセットの詳細については、下記の発表資料を参照してください。

- 鈴木正敏. 4択クイズを題材にした多肢選択式日本語質問応答データセットの構築. 言語処理学会第30回年次大会 (NLP2024) 併設ワークショップ 日本語言語資源の構築と利用性の向上 (JLR2024), 2024. \[[PDF](https://jedworkshop.github.io/JLR2024/materials/a-1.pdf)\]

データセットは Hugging Face Hub にて公開しています。

- https://huggingface.co/datasets/tohoku-nlp/abc-multiple-choice

このリポジトリでは、評価実験に用いたスクリプト群を管理しています。

以降の内容は、本データセットを用いた評価実験の手順です。


## 実験環境

- Python 3.10.13
- GPU: NVIDIA GeForce RTX 3090 (24GB) x 4
- CUDA Version: 12.1


## インストール

```sh
git clone --recursive https://github.com/cl-tohoku/abc-multiple-choice.git
cd abc-multiple-choice

pip install -e lm-evaluation-harness
pip install -e llm-jp-eval
pip install sentencepiece
```

実験を行った環境で上記手順によりインストールされたパッケージの一覧が [`requirements.txt`](./requirements.txt) に記述されています。
実験結果の再現を試みたい場合などに、必要に応じて参照してください。


## `lm-evaluation-harness` を用いた対数尤度による評価

### 準備

```sh
mkdir lm-evaluation-harness/lm_eval/tasks/abc_multiple_choice
cp patch/lm-evaluation-harness/abc_multiple_choice_0.2.1.yaml lm-evaluation-harness/lm_eval/tasks/abc_multiple_choice
```

### プロンプトの確認

```sh
TASK="abc_multiple_choice_0.2.1"

python lm-evaluation-harness/scripts/write_out.py \
  --output_base_path work/lm-evaluation-harness/prompts \
  --tasks $TASK \
  --sets test \
  --num_fewshot 4 \
  --num_examples 10
```

### 評価の実行

```sh
TASK="abc_multiple_choice_0.2.1"

MODEL="cyberagent/calm2-7b"
# MODEL="elyza/ELYZA-japanese-Llama-2-7b"
# MODEL="elyza/ELYZA-japanese-Llama-2-7b-fast"
# MODEL="elyza/ELYZA-japanese-Llama-2-13b"
# MODEL="elyza/ELYZA-japanese-Llama-2-13b-fast"
# MODEL="llm-jp/llm-jp-13b-v1.0"
# MODEL="stabilityai/japanese-stablelm-base-beta-7b"
# MODEL="stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
# MODEL="stabilityai/japanese-stablelm-base-gamma-7b"
# MODEL="stockmark/stockmark-13b"
# MODEL="tokyotech-llm/Swallow-7b-hf"
# MODEL="tokyotech-llm/Swallow-13b-hf"
# MODEL="matsuo-lab/weblab-10b"
# MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="meta-llama/Llama-2-13b-hf"
# MODEL="mistralai/Mistral-7B-v0.1"

lm_eval --model hf \
  --tasks $TASK \
  --model_args "pretrained=$MODEL,parallelize=True" \
  --num_fewshot 4 \
  --device cuda \
  --output_path "work/lm-evaluation-harness/logs/$TASK/fewshot/$MODEL" \
  --log_samples
```

### 結果の集計

```sh
MODELS=(
  "cyberagent/calm2-7b"
  "elyza/ELYZA-japanese-Llama-2-7b"
  "elyza/ELYZA-japanese-Llama-2-7b-fast"
  "elyza/ELYZA-japanese-Llama-2-13b"
  "elyza/ELYZA-japanese-Llama-2-13b-fast"
  "llm-jp/llm-jp-13b-v1.0"
  "stabilityai/japanese-stablelm-base-beta-7b"
  "stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
  "stabilityai/japanese-stablelm-base-gamma-7b"
  "stockmark/stockmark-13b"
  "tokyotech-llm/Swallow-7b-hf"
  "tokyotech-llm/Swallow-13b-hf"
  "matsuo-lab/weblab-10b"
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "mistralai/Mistral-7B-v0.1"
)

python aggregate_results.py \
  --dataset_name tohoku-nlp/abc-multiple-choice \
  --dataset_split_name test \
  --type lm-evaluation-harness \
  --results_root_dir work/lm-evaluation-harness/logs/abc_multiple_choice_0.2.1/fewshot \
  --model_names $MODELS \
  --invalid_qids_file invalid_qids.txt \
  --output_file work/lm-evaluation-harness/logs/abc_multiple_choice_0.2.1/fewshot/results.tsv
# model_name      acc     acc_filtered
# cyberagent/calm2-7b     0.587   0.594
# elyza/ELYZA-japanese-Llama-2-7b 0.407   0.421
# elyza/ELYZA-japanese-Llama-2-7b-fast    0.427   0.437
# elyza/ELYZA-japanese-Llama-2-13b        0.571   0.581
# elyza/ELYZA-japanese-Llama-2-13b-fast   0.558   0.579
# llm-jp/llm-jp-13b-v1.0  0.529   0.520
# stabilityai/japanese-stablelm-base-beta-7b      0.464   0.482
# stabilityai/japanese-stablelm-base-ja_vocab-beta-7b     0.458   0.470
# stabilityai/japanese-stablelm-base-gamma-7b     0.671   0.688
# stockmark/stockmark-13b 0.740   0.739
# tokyotech-llm/Swallow-7b-hf     0.713   0.706
# tokyotech-llm/Swallow-13b-hf    0.749   0.746
# matsuo-lab/weblab-10b   0.389   0.401
# meta-llama/Llama-2-7b-hf        0.384   0.388
# meta-llama/Llama-2-13b-hf       0.473   0.492
# mistralai/Mistral-7B-v0.1       0.362   0.368
```


## `llm-jp-eval` を用いた選択肢の出力による評価

### 準備

```sh
cp llm-jp-eval/configs/config_template.yaml llm-jp-eval/configs/config.yaml

patch -p1 -d llm-jp-eval < patch/llm-jp-eval/diff.patch
cp patch/llm-jp-eval/abc_multiple_choice.py llm-jp-eval/src/llm_jp_eval/datasets/
cp patch/llm-jp-eval/abc_multiple_choice_numberless.py llm-jp-eval/src/llm_jp_eval/datasets/
```

### データセットの作成

```sh
DATASET_NAME="abc_multiple_choice"               # 選択肢の番号を出力して解答
# DATASET_NAME="abc_multiple_choice_numberless"  # 選択肢の語句を出力して解答

python llm-jp-eval/scripts/preprocess_dataset.py \
  --dataset-name $DATASET_NAME \
  --output-dir work/llm-jp-eval/datasets
```

### 評価の実行

```sh
TASK="abc_multiple_choice"               # 選択肢の番号を出力して解答
# TASK="abc_multiple_choice_numberless"  # 選択肢の語句を出力して解答

MODEL="cyberagent/calm2-7b"
# MODEL="elyza/ELYZA-japanese-Llama-2-7b"
# MODEL="elyza/ELYZA-japanese-Llama-2-7b-fast"
# MODEL="elyza/ELYZA-japanese-Llama-2-13b"
# MODEL="elyza/ELYZA-japanese-Llama-2-13b-fast"
# MODEL="llm-jp/llm-jp-13b-v1.0"
# MODEL="stabilityai/japanese-stablelm-base-beta-7b"
# MODEL="stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
# MODEL="stabilityai/japanese-stablelm-base-gamma-7b"
# MODEL="stockmark/stockmark-13b"
# MODEL="tokyotech-llm/Swallow-7b-hf"
# MODEL="tokyotech-llm/Swallow-13b-hf"
# MODEL="matsuo-lab/weblab-10b"
# MODEL="meta-llama/Llama-2-7b-hf"
# MODEL="meta-llama/Llama-2-13b-hf"
# MODEL="mistralai/Mistral-7B-v0.1"

CUDA_VISIBLE_DEVICES=0 python llm-jp-eval/scripts/evaluate_llm.py -cn config.yaml \
  model.pretrained_model_name_or_path=$MODEL \
  tokenizer.pretrained_model_name_or_path=$MODEL \
  dataset_dir=work/llm-jp-eval/datasets/1.2.0/evaluation/test \
  target_dataset="[$TASK]" \
  metainfo.num_few_shots=4 \
  metainfo.max_num_samples=-1 \
  log_dir=work/llm-jp-eval/logs/$TASK/fewshot/$MODEL
```

### 結果の集計

```sh
MODELS=(
  "cyberagent/calm2-7b"
  "elyza/ELYZA-japanese-Llama-2-7b"
  "elyza/ELYZA-japanese-Llama-2-7b-fast"
  "elyza/ELYZA-japanese-Llama-2-13b"
  "elyza/ELYZA-japanese-Llama-2-13b-fast"
  "llm-jp/llm-jp-13b-v1.0"
  "stabilityai/japanese-stablelm-base-beta-7b"
  "stabilityai/japanese-stablelm-base-ja_vocab-beta-7b"
  "stabilityai/japanese-stablelm-base-gamma-7b"
  "stockmark/stockmark-13b"
  "tokyotech-llm/Swallow-7b-hf"
  "tokyotech-llm/Swallow-13b-hf"
  "matsuo-lab/weblab-10b"
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "mistralai/Mistral-7B-v0.1"
)

# 選択肢の番号を出力して解答
python aggregate_results.py \
  --dataset_name tohoku-nlp/abc-multiple-choice \
  --dataset_split_name test \
  --type llm-jp-eval \
  --results_root_dir work/llm-jp-eval/logs/abc_multiple_choice/fewshot \
  --model_names $MODELS \
  --invalid_qids_file invalid_qids.txt \
  --output_file work/llm-jp-eval/logs/abc_multiple_choice/fewshot/results.tsv
# model_name      acc     acc_filtered
# cyberagent/calm2-7b     0.227   0.234
# elyza/ELYZA-japanese-Llama-2-7b 0.338   0.353
# elyza/ELYZA-japanese-Llama-2-7b-fast    0.351   0.363
# elyza/ELYZA-japanese-Llama-2-13b        0.493   0.503
# elyza/ELYZA-japanese-Llama-2-13b-fast   0.473   0.482
# llm-jp/llm-jp-13b-v1.0  0.224   0.231
# stabilityai/japanese-stablelm-base-beta-7b      0.324   0.330
# stabilityai/japanese-stablelm-base-ja_vocab-beta-7b     0.249   0.249
# stabilityai/japanese-stablelm-base-gamma-7b     0.596   0.609
# stockmark/stockmark-13b 0.271   0.272
# tokyotech-llm/Swallow-7b-hf     0.433   0.439
# tokyotech-llm/Swallow-13b-hf    0.593   0.614
# matsuo-lab/weblab-10b   0.244   0.254
# meta-llama/Llama-2-7b-hf        0.296   0.294
# meta-llama/Llama-2-13b-hf       0.373   0.386
# mistralai/Mistral-7B-v0.1       0.340   0.358

# 選択肢の語句を出力して解答
python aggregate_results.py \
  --dataset_name tohoku-nlp/abc-multiple-choice \
  --dataset_split_name test \
  --type llm-jp-eval \
  --results_root_dir work/llm-jp-eval/logs/abc_multiple_choice_numberless/fewshot \
  --model_names $MODELS \
  --invalid_qids_file invalid_qids.txt \
  --output_file work/llm-jp-eval/logs/abc_multiple_choice_numberless/fewshot/results.tsv
# model_name      acc     acc_filtered
# cyberagent/calm2-7b     0.531   0.536
# elyza/ELYZA-japanese-Llama-2-7b 0.382   0.386
# elyza/ELYZA-japanese-Llama-2-7b-fast    0.360   0.368
# elyza/ELYZA-japanese-Llama-2-13b        0.562   0.569
# elyza/ELYZA-japanese-Llama-2-13b-fast   0.460   0.477
# llm-jp/llm-jp-13b-v1.0  0.436   0.442
# stabilityai/japanese-stablelm-base-beta-7b      0.442   0.457
# stabilityai/japanese-stablelm-base-ja_vocab-beta-7b     0.393   0.406
# stabilityai/japanese-stablelm-base-gamma-7b     0.613   0.632
# stockmark/stockmark-13b 0.000   0.000
# tokyotech-llm/Swallow-7b-hf     0.580   0.591
# tokyotech-llm/Swallow-13b-hf    0.704   0.695
# matsuo-lab/weblab-10b   0.360   0.363
# meta-llama/Llama-2-7b-hf        0.289   0.302
# meta-llama/Llama-2-13b-hf       0.382   0.396
# mistralai/Mistral-7B-v0.1       0.342   0.343
```


## ライセンス

- 本データセットのクイズ問題の著作権は [abc/EQIDEN 実行委員会](https://abc-dive.com/portal/) に帰属します。
- 本データセットは研究目的での利用許諾を得ているものです。商用目的での利用は不可とします。
