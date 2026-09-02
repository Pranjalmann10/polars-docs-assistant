"""Stage 9b: train the QLoRA adapter for arm C's generator.

Loads Qwen2.5-1.5B-Instruct in 4-bit (bitsandbytes nf4) and trains a LoRA
adapter on the distilled dataset from `src/train/build_sft_data.py`
(data/sft_train.jsonl). The training format exactly mirrors
`src/generator/model.py::generate`'s RAG prompt (RAG_SYSTEM_PROMPT + the same
"Context:\n...\n\nQuestion: ..." user turn), so the adapter learns the answer
format the evaluation harness will actually invoke it with. Only the
assistant's answer tokens are trained on — the prompt is masked with -100 so
the loss doesn't reward memorizing the context or question.

Saved to models/qwen2.5-1.5b-polars-qlora/, with adapter_config.json's
base_model_name_or_path set to BASE_MODEL_NAME so
generator/model.py::_verify_adapter_base_match can catch a mismatch.
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.generator.model import BASE_MODEL_NAME, RAG_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_DATA_PATH = REPO_ROOT / "data" / "sft_train.jsonl"
ADAPTER_OUT = REPO_ROOT / "models" / "qwen2.5-1.5b-polars-qlora"

MAX_LEN = 1024


def _load_records() -> list[dict]:
    records = []
    for line in SFT_DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _build_example(tokenizer, record: dict) -> dict:
    user = f"Context:\n{record['context']}\n\nQuestion: {record['question']}"
    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    prompt_ids = list(prompt_ids)
    answer_ids = tokenizer(
        record["answer"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:MAX_LEN]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = _load_records()
    print(f"[train] {len(records)} SFT examples from {SFT_DATA_PATH}")
    examples = [_build_example(tokenizer, r) for r in records]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to=[],
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(examples),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    # PEFT writes base_model_name_or_path from the loaded model's name_or_path,
    # which already matches BASE_MODEL_NAME here, but assert it explicitly so a
    # future change to this script can't silently break the safety check in
    # generator/model.py::_verify_adapter_base_match.
    config_path = ADAPTER_OUT / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == BASE_MODEL_NAME
    print(f"[train] adapter saved to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
