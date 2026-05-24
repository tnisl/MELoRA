#!/usr/bin/env python
# coding=utf-8
"""Fine-tune PhoBERT on lizNguyen235/vietnamese-nli-phobert with LoRA/MELoRA.
Adapted from MELoRA run_glue_lora.py.
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    LoraConfig,
    MELoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

check_min_version("4.29.0")
require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)

LABEL_LIST = ["entailment", "neutral", "contradiction"]  # dataset IDs: 0, 1, 2
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="lizNguyen235/vietnamese-nli-phobert",
        metadata={"help": "HuggingFace dataset name."},
    )
    dataset_config_name: Optional[str] = field(default=None)
    premise_column: str = field(
        default="premise_seg",
        metadata={"help": "Use premise_seg for PhoBERT; use premise for raw text."},
    )
    hypothesis_column: str = field(
        default="hypothesis_seg",
        metadata={"help": "Use hypothesis_seg for PhoBERT; use hypothesis for raw text."},
    )
    label_column: str = field(default="label")
    wandb_project: Optional[str] = field(default="")
    wandb_watch: Optional[str] = field(default="")
    wandb_log_model: Optional[str] = field(default="")
    max_seq_length: int = field(default=256)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="vinai/phobert-base-v2")
    lora_path: Optional[str] = field(default=None)
    l_num: Optional[int] = field(default=None)
    mode: str = field(default="base", metadata={"help": "base = LoRA, me = MELoRA."})
    config_name: Optional[str] = field(default=None)
    rank: List[int] = field(default_factory=lambda: [8])
    lora_alpha: List[int] = field(default_factory=lambda: [16])
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["query", "value"])
    lora_dropout: Optional[float] = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_task_type: str = field(default="SEQ_CLS")
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)


def print_lora_parameters(model):
    trainable_params = 0
    lora_params = 0
    all_param = 0
    for n, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if "original_module" in n:
            continue
        if param.requires_grad:
            trainable_params += num_params
            if "lora_" in n:
                lora_params += num_params
            else:
                print("trainable non-lora:", n)
    print(
        f"lora params: {lora_params:,d} || trainable params: {trainable_params:,d} "
        f"|| all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def get_classifier_parameters(peft_model):
    # Roberta/PhoBERT sequence-classification head after get_peft_model.
    base = peft_model.base_model.model
    if hasattr(base, "classifier"):
        return list(base.classifier.parameters())
    return []


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.l_num is not None:
        model_args.rank = [model_args.rank[0]] * model_args.l_num
        model_args.lora_alpha = [model_args.lora_alpha[0]] * model_args.l_num

    if data_args.wandb_project:
        os.environ["WANDB_PROJECT"] = data_args.wandb_project
    if data_args.wandb_watch:
        os.environ["WANDB_WATCH"] = data_args.wandb_watch
    if data_args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = data_args.wandb_log_model

    send_example_telemetry("run_vinli_lora", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, "
        f"fp16: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}.")

    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    for split_name, ds in raw_datasets.items():
        missing = {data_args.premise_column, data_args.hypothesis_column, data_args.label_column} - set(ds.column_names)
        if missing:
            raise ValueError(f"Split {split_name} missing columns: {missing}. Available: {ds.column_names}")

    num_labels = 3
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if "me" in model_args.mode:
        print("*** MELoRA ***")
        peft_config = MELoraConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            mode=model_args.mode,
            task_type=model_args.lora_task_type,
        )
    elif "base" in model_args.mode:
        print("*** LoRA ***")
        peft_config = LoraConfig(
            r=model_args.rank[0],
            lora_alpha=model_args.lora_alpha[0],
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type=model_args.lora_task_type,
        )
    else:
        raise ValueError(f"Unknown mode {model_args.mode}; use base or me")

    model = get_peft_model(model, peft_config)
    print_lora_parameters(model)

    if model_args.lora_path is not None:
        adapter_path = os.path.join(model_args.lora_path, "adapter_model.bin")
        print(f"*** Load adapter weights from {adapter_path} ***")
        adapters_weights = torch.load(adapter_path, map_location=model.device)
        filtered_dict = {k: v for k, v in adapters_weights.items() if "classifier" not in k}
        set_peft_model_state_dict(model, filtered_dict)
        del adapters_weights

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        result = tokenizer(
            examples[data_args.premise_column],
            examples[data_args.hypothesis_column],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )
        result["label"] = examples[data_args.label_column]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
        macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
        return {"accuracy": acc, "macro_f1": macro_f1}

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    head_params = get_classifier_parameters(model)
    if head_params:
        head_param_ids = set(map(id, head_params))
        base_params = [p for p in model.parameters() if id(p) not in head_param_ids and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params},
                {"params": head_params, "lr": training_args.learning_rate / 2},
            ],
            lr=training_args.learning_rate,
        )
        optimizers = (optimizer, None)
    else:
        optimizers = (None, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    # torch.compile sometimes breaks PEFT/debugging; enable manually if needed.
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        pred_ds = predict_dataset.remove_columns("label") if "label" in predict_dataset.column_names else predict_dataset
        predictions = trainer.predict(pred_ds, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results_vinli.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\tprediction_id\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{ID2LABEL[int(item)]}\t{int(item)}\n")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
