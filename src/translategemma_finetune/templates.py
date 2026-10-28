from collections.abc import Iterable, Mapping
from typing import Any
import json
import re
from transformers import PreTrainedTokenizer

LANGUAGES_BLOCK_END_RE = re.compile(r"\A\{%- set languages = \{\n.*?(?=\n\}\n-%\})", re.DOTALL)


def parse_extra_languages(values: Iterable[str]) -> dict[str, str]:
    """Parse ``CODE=Name`` entries, normalizing codes the way the chat template does."""
    languages: dict[str, str] = {}
    for value in values:
        code, separator, name = value.partition("=")
        code = code.strip().replace("_", "-")
        name = name.strip()
        if not separator or not code or not name:
            raise ValueError(f"Expected CODE=Name, got {value!r}")
        languages[code] = name
    return languages


def add_languages_to_chat_template(chat_template: str, languages: dict[str, str]) -> str:
    """Append languages to the ``languages`` map at the top of the TranslateGemma template.

    Codes already in the map are kept as they are, so applying this twice is a no-op.
    """
    match = LANGUAGES_BLOCK_END_RE.match(chat_template)
    if not match:
        if not languages:
            return chat_template
        raise ValueError("Could not find the `languages` block in the chat template.")

    entries = "".join(
        f"\n    {json.dumps(code)}: {json.dumps(name, ensure_ascii=False)},"
        for code, name in languages.items()
        if f"\n    {json.dumps(code)}:" not in match.group()
    )
    return chat_template[: match.end()] + entries + chat_template[match.end() :]


def row_lang_codes(
    examples: Mapping[str, list[Any]],
    count: int,
    source_lang_code: str,
    target_lang_code: str,
) -> tuple[list[str], list[str]]:
    """Per-row codes from ``source_lang_code``/``target_lang_code`` columns, else the defaults.

    Mixed-direction data (see ``mix_data.py``) stores the direction of every row.
    """
    return (
        list(examples.get("source_lang_code") or [source_lang_code] * count),
        list(examples.get("target_lang_code") or [target_lang_code] * count),
    )


def raw_prompt_prefix(tokenizer: PreTrainedTokenizer | None) -> str:
    """Return the ``<bos>`` that the chat template would add, for the raw prompt format.

    Prompts are tokenized with ``add_special_tokens=False``, so the text must carry it.
    """
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    return getattr(tokenizer, "bos_token", None) or ""


def create_chat_messages(
    source_text: str,
    target_text: str | None = None,
    source_lang_code: str = "fr",
    target_lang_code: str = "mos_Latn",
) -> list[dict[str, Any]]:
    """
    Create chat messages for both training and inference.

    Args:
        source_text: Source text to translate
        target_text: Target translation (optional, for training)
        model_id: Model identifier to determine template type
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        List of chat messages in the appropriate format
    """
    if not source_lang_code or not target_lang_code:
        raise ValueError("source_lang_code and target_lang_code must be provided")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang_code,
                    "target_lang_code": target_lang_code,
                    "text": source_text,
                }
            ],
        }
    ]

    if target_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": target_text,
            }
        )

    return messages


def format_for_prediction(
    examples: dict[str, list[str]],
    source_lang_code: str = "fr",
    target_lang_code: str = "mos_Latn",
    source_field: str = "french",
    tokenizer: PreTrainedTokenizer | None = None,
    use_chat_template: bool = False,
) -> dict[str, list[str]]:
    texts = [
        format_single_for_prediction(
            source_text=source,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        for source in examples[source_field]
    ]
    return {"text": texts}


def format_single_for_prediction(
    source_text: str,
    source_lang_code: str = "fr",
    target_lang_code: str = "mos_Latn",
    tokenizer: PreTrainedTokenizer | None = None,
    use_chat_template: bool = False,
) -> str:
    if use_chat_template and tokenizer is None:
        raise ValueError("tokenizer must be provided when use_chat_template=True")

    messages = create_chat_messages(
        source_text,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    )

    if use_chat_template and tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )  # ty:ignore[invalid-return-type]

    json_str = json.dumps(messages[0]["content"], ensure_ascii=False)
    return f"{raw_prompt_prefix(tokenizer)}user\n{json_str}\nmodel\n"


def format_for_training(
    examples: dict[str, list[str]],
    source_lang_code: str = "fr",
    target_lang_code: str = "mos_Latn",
    source_field: str = "french",
    target_field: str = "moore",
    tokenizer: PreTrainedTokenizer | None = None,
    use_chat_template: bool = False,
) -> dict[str, list[str]]:
    if use_chat_template and tokenizer is None:
        raise ValueError("tokenizer must be provided when use_chat_template=True")

    sources, targets = examples[source_field], examples[target_field]
    source_codes, target_codes = row_lang_codes(
        examples, len(sources), source_lang_code, target_lang_code
    )
    texts = []
    for source, target, row_source_code, row_target_code in zip(
        sources, targets, source_codes, target_codes, strict=True
    ):
        messages = create_chat_messages(
            source,
            target_text=target,
            source_lang_code=row_source_code,
            target_lang_code=row_target_code,
        )

        if use_chat_template and tokenizer is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            json_str = json.dumps(messages[0]["content"], ensure_ascii=False)
            prefix = raw_prompt_prefix(tokenizer)
            text = f"{prefix}user\n{json_str}\nmodel\n{messages[1]['content']}"

        texts.append(text)

    return {"text": texts}
