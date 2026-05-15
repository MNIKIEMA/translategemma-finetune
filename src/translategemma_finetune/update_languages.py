from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANGUAGES_PATH = REPO_ROOT / "src" / "translategemma_finetune" / "languages.json"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "translategemma_template.jinja"
NEW_TEMPLATE_PATH = REPO_ROOT / "chat_template.jinja"
LANGUAGES_BLOCK_RE = re.compile(r"\A\{%- set languages = \{\n.*?\n\}\n-%\}", re.DOTALL)


def normalize_code(code: str) -> str:
    return code.strip().replace("_", "-")


def load_languages(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return _validate_language_map(value, path)


def _validate_language_map(value: Mapping[object, object], path: Path) -> dict[str, str]:
    languages: dict[str, str] = {}
    for code, name in value.items():
        if not isinstance(code, str) or not isinstance(name, str):
            raise ValueError(f"{path} language map must contain only string keys and values")
        languages[normalize_code(code)] = name.strip()
    return languages


def render_json_map(languages: Mapping[str, str]) -> str:
    return (
        json.dumps(
            dict(_sorted_languages(languages)),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def render_jinja_languages_block(languages: Mapping[str, str]) -> str:
    lines = ["{%- set languages = {"]
    for code, name in _sorted_languages(languages):
        lines.append(f'    "{code}": "{_escape(name)}",')
    lines.append("}")
    lines.append("-%}")
    return "\n".join(lines)


def replace_languages_block(template_text: str, rendered_block: str, path: Path) -> str:
    match = LANGUAGES_BLOCK_RE.match(template_text)
    if not match:
        raise ValueError(f"Could not find the Jinja `languages` block in {path}")
    return rendered_block + template_text[match.end() :].lstrip("\n")


def _sorted_languages(languages: Mapping[str, str]) -> Iterable[tuple[str, str]]:
    return sorted(languages.items(), key=lambda item: item[0].casefold())


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def update_language_files(
    new_languages: Mapping[str, str],
    languages_path: Path = DEFAULT_LANGUAGES_PATH,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
    new_template_path: Path = NEW_TEMPLATE_PATH,
) -> dict[str, str]:
    languages = load_languages(languages_path)
    normalized_new_languages = {
        normalize_code(code): name.strip()
        for code, name in new_languages.items()
        if normalize_code(code) and name.strip()
    }
    if not normalized_new_languages:
        raise ValueError("Provide at least one non-empty language code and name")

    languages.update(normalized_new_languages)

    template_text = template_path.read_text(encoding="utf-8")
    rendered_json_map = render_json_map(languages)
    rendered_block = render_jinja_languages_block(languages)
    updated_template = replace_languages_block(template_text, rendered_block, template_path)

    languages_path.write_text(rendered_json_map, encoding="utf-8")  #TODO: Do we need to write this file?
    new_template_path.write_text(updated_template, encoding="utf-8")

    return normalized_new_languages


def parse_language_values(values: Iterable[str]) -> dict[str, str]:
    languages: dict[str, str] = {}
    for value in values:
        code, separator, name = value.partition("=")
        if not separator:
            raise ValueError(f"Expected CODE=Name, got {value!r}")
        code = normalize_code(code)
        name = name.strip()
        if not code or not name:
            raise ValueError(f"Expected non-empty CODE and Name, got {value!r}")
        languages[code] = name
    return languages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add or update language names in languages.json and chat_template.jinja."
    )
    parser.add_argument(
        "language",
        nargs="+",
        help='Language entries as CODE=Name, for example "ff-Adlm=Fulah".',
    )
    parser.add_argument(
        "--languages",
        type=Path,
        default=DEFAULT_LANGUAGES_PATH,
        help=f"Path to languages.json. Defaults to {DEFAULT_LANGUAGES_PATH}.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help=f"Path to the source template. Defaults to {DEFAULT_TEMPLATE_PATH}.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        updated = update_language_files(
            parse_language_values(args.language),
            languages_path=args.languages,
            template_path=args.template,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for code, name in updated.items():
        print(f'Updated "{code}": "{name}"')


if __name__ == "__main__":
    main()
