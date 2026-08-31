"""Stage 7: the answer generator — base Qwen2.5-1.5B-Instruct, optionally with a
QLoRA adapter layered on top.

No adapter exists for this project yet (see README "Open questions"). This
module is written so that arm A/B of the evaluation (base model, no adapter)
work today, and arm C (base + adapter) activates automatically the moment
ADAPTER_PATH is set to a real checkpoint — no code changes required.

The adapter is NOT a model. It's a few MB of low-rank weight deltas that only
mean anything loaded on top of the exact base they were trained against. We
verify that match before doing anything else, because a mismatch either fails
on a shape error or, worse, loads fine and silently produces garbage.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip() or None

RAG_SYSTEM_PROMPT = (
    "You are a Polars documentation assistant. Answer the user's question using "
    "ONLY the provided context passages. Include a short code example in a fenced "
    "python block when relevant. Cite which source doc the answer came from. "
    "If the context does not contain the answer, say so explicitly instead of "
    "guessing."
)

NO_RAG_SYSTEM_PROMPT = (
    "You are a Polars documentation assistant. Answer the user's Polars question "
    "as accurately as you can."
)


@dataclass(frozen=True)
class GeneratorConfig:
    base_model_name: str = BASE_MODEL_NAME
    adapter_path: str | None = ADAPTER_PATH
    max_new_tokens: int = 512
    temperature: float = 0.2


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def _verify_adapter_base_match(adapter_path: str, base_model_name: str) -> None:
    """Fail loudly and early if the adapter was trained against a different base.

    A silent mismatch is worse than a crash: shapes can coincidentally line up
    for some layers and produce fluent-looking, wrong output.
    """
    import json
    from pathlib import Path

    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found at {adapter_path}. "
            "This does not look like a PEFT/LoRA adapter directory."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trained_base = config.get("base_model_name_or_path")
    if trained_base and trained_base != base_model_name:
        raise ValueError(
            f"Adapter at {adapter_path} was trained against base model "
            f"'{trained_base}', but this generator is configured to load base "
            f"model '{base_model_name}'. Loading it anyway would either fail on "
            "shape mismatch or silently produce garbage. Fix BASE_MODEL_NAME or "
            "ADAPTER_PATH."
        )


@lru_cache(maxsize=1)
def _load(config: GeneratorConfig = GeneratorConfig()):
    device = _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)

    if config.adapter_path:
        _verify_adapter_base_match(config.adapter_path, config.base_model_name)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
        print(f"[generator] loaded adapter from {config.adapter_path}")
    else:
        print("[generator] no ADAPTER_PATH set — running base model only")

    model.eval()
    return tokenizer, model, device


def generate(
    question: str,
    context: str | None = None,
    config: GeneratorConfig = GeneratorConfig(),
) -> str:
    """Generate an answer. If `context` is None, this is arm A (no retrieval).
    If `context` is provided, this is arm B/C (RAG), with or without the adapter
    depending on `config.adapter_path`."""
    tokenizer, model, device = _load(config)

    if context:
        system = RAG_SYSTEM_PROMPT
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = NO_RAG_SYSTEM_PROMPT
        user = question

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(generate("How do I filter rows in a Polars DataFrame?"))
