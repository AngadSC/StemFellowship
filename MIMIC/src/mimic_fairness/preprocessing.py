from transformers import AutoTokenizer


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_text(text: str, tokenizer, max_length: int) -> dict:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
