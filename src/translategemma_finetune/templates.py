from typing import Any


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
