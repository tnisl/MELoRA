# Hướng dẫn chạy MELoRA với PhoBERT trên Vietnamese NLI

README này hướng dẫn setup môi trường và chạy thí nghiệm LoRA/MELoRA trên repo [`tnisl/MELoRA`](https://github.com/tnisl/MELoRA), dựa trên notebook `vli-melora.ipynb`.

## 1. Clone repo

```bash
git clone https://github.com/tnisl/MELoRA.git
cd MELoRA
```

Repo đã có script `run_vinli_lora.py` để chạy Vietnamese NLI.

## 2. Tạo môi trường Python 3.10 bằng `uv`

> Khuyến nghị dùng Python 3.10. Không dùng Python 3.12 nếu requirements đang pin `torch==2.0.1`, vì phiên bản này không có wheel cho `cp312`.

```bash
pip install uv

rm -rf .venv uv.lock

uv python install 3.10
uv venv .venv --python 3.10

source .venv/bin/activate

python --version
which python
```

Kỳ vọng:

```text
Python 3.10.x
.../MELoRA/.venv/bin/python
```

Nếu `which python` vẫn trỏ tới `/usr/bin/python`, chạy trực tiếp bằng path:

```bash
./.venv/bin/python --version
```

## 3. Cài dependencies

Cài PyTorch và các package trong repo:

```bash
uv pip install --python .venv/bin/python torch==2.0.1
uv pip install --python .venv/bin/python -r requirements.txt
```

Cài PEFT custom trong repo MELoRA:

```bash
cd peft-0.5.0
uv pip install --python ../.venv/bin/python -e .
cd ..
```

Kiểm tra `MELoraConfig`:

```bash
./.venv/bin/python -c "from peft import MELoraConfig; print(MELoraConfig)"
```

Nếu lệnh trên lỗi:

```text
ImportError: cannot import name 'MELoraConfig' from 'peft'
```

thì bạn đang dùng PEFT official từ pip thay vì PEFT custom của repo. Cài lại:

```bash
uv pip uninstall --python .venv/bin/python -y peft
cd peft-0.5.0
uv pip install --python ../.venv/bin/python -e .
cd ..
```

## 4. Tắt W&B

Có thể tắt bằng biến môi trường:

```bash
export WANDB_DISABLED=true
```

hoặc thêm vào command:

```bash
--report_to none
```

## 5. Dataset và model

Dataset dùng trong notebook:

```text
lizNguyen235/vietnamese-nli-phobert
```

Các cột input:

```text
premise_seg
hypothesis_seg
label
```

Model dùng trong notebook:

```text
vinai/phobert-base
```

Nếu muốn dùng PhoBERT v2, đổi thành:

```text
vinai/phobert-base-v2
```

## 6. Chạy LoRA baseline

Command debug với 10k train samples:

```bash
WANDB_DISABLED=true uv run --python .venv/bin/python python run_vinli_lora.py \
  --model_name_or_path vinai/phobert-base \
  --dataset_name lizNguyen235/vietnamese-nli-phobert \
  --premise_column premise_seg \
  --hypothesis_column hypothesis_seg \
  --label_column label \
  --mode base \
  --rank 8 \
  --lora_alpha 16 \
  --target_modules query value \
  --lora_dropout 0.05 \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --save_steps 2000 \
  --logging_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --output_dir outputs/phobert-vinli-lora \
  --overwrite_output_dir \
  --fp16 \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --max_predict_samples 1000 \
  --report_to none
```

Ý nghĩa:

```text
--mode base      chạy LoRA thường
--rank 8         rank LoRA
--target_modules query value
                 gắn LoRA vào attention query/value
```

## 7. Chạy MELoRA

Command debug với 10k train samples:

```bash
WANDB_DISABLED=true uv run --python .venv/bin/python python run_vinli_lora.py \
  --model_name_or_path vinai/phobert-base \
  --dataset_name lizNguyen235/vietnamese-nli-phobert \
  --premise_column premise_seg \
  --hypothesis_column hypothesis_seg \
  --label_column label \
  --mode me \
  --l_num 2 \
  --rank 8 \
  --lora_alpha 16 \
  --target_modules query value \
  --lora_dropout 0.05 \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --save_steps 2000 \
  --logging_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 3 \
  --max_seq_length 256 \
  --output_dir outputs/phobert-vinli-melora \
  --overwrite_output_dir \
  --fp16 \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --max_train_samples 10000 \
  --max_eval_samples 1000 \
  --max_predict_samples 1000 \
  --report_to none
```

Ý nghĩa:

```text
--mode me        chạy MELoRA
--l_num 2        số lượng mini-LoRA
--rank 8         rank của từng mini-LoRA theo config repo
```

