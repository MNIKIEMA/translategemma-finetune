"""Load translation data as rows with one shape, whatever the source.

Every row has ``source_text``, ``target_text``, ``source_lang_code`` and ``target_lang_code``.
Training and evaluation read only these columns, so the direction of each row travels with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SOURCE_COLUMN = "source_text"
TARGET_COLUMN = "target_text"
MOORE_CODE = "mos_Latn"
FRENCH_CODE = "fr"
DEFAULT_MIX_DIR = "data/fra-mos-mix"

# Column pairs recognized in datasets that do not use the names above, with their direction.
KNOWN_TEXT_COLUMNS = [
    ("french", "moore", FRENCH_CODE, MOORE_CODE),
]

BOUQUET = "facebook/bouquet"
BOUQUET_CODES = {"fra_Latn": FRENCH_CODE, "mos_Latn": MOORE_CODE}
MOORE_WEB = "madoss/moore-web-parallel"

DIRECTIONS = {
    "fr-mos": (FRENCH_CODE, MOORE_CODE),
    "mos-fr": (MOORE_CODE, FRENCH_CODE),
}


def bouquet_files(split: str) -> list[str]:
    """Sentence-level BOUQuET ``split`` (dev or test), French -> Mooré and Mooré -> French."""
    return [
        f"benchmark_data/sentence_level/{split}/{pair}.parquet"
        for pair in ("fra_Latn-mos_Latn", "mos_Latn-fra_Latn")
    ]


def to_translation_rows(dataset: Any) -> Any:
    """Rename a dataset's text columns to the shared names and add its direction if missing."""
    columns = set(dataset.column_names)
    if {SOURCE_COLUMN, TARGET_COLUMN} <= columns:
        if not {"source_lang_code", "target_lang_code"} <= columns:
            raise ValueError(
                "Rows with source_text/target_text need source_lang_code/target_lang_code."
            )
        return dataset
    for source, target, source_code, target_code in KNOWN_TEXT_COLUMNS:
        if {source, target} <= columns:
            dataset = dataset.rename_columns({source: SOURCE_COLUMN, target: TARGET_COLUMN})
            dataset = dataset.add_column("source_lang_code", [source_code] * len(dataset))
            return dataset.add_column("target_lang_code", [target_code] * len(dataset))
    raise ValueError(
        f"Unrecognized columns {sorted(columns)}: expected source_text/target_text "
        "with language codes, or french/moore."
    )


def load_translation_split(data: str, split: str) -> Any | None:
    """Load ``split`` from a mix directory (``<split>.jsonl``), a data file or a Hub dataset.

    Returns None when a mix directory or Hub dataset has no such split.
    """
    from datasets import get_dataset_split_names, load_dataset

    path = Path(data)
    if path.is_dir():
        split_file = path / f"{split}.jsonl"
        if not split_file.exists():
            return None
        dataset = load_dataset("json", data_files=str(split_file), split="train")
    elif path.is_file():
        file_type = "parquet" if path.suffix == ".parquet" else "json"
        dataset = load_dataset(file_type, data_files=str(path), split="train")
    elif path.is_absolute() or data.startswith(".") or data == DEFAULT_MIX_DIR:
        raise FileNotFoundError(f"{data} does not exist. Build the mix with `translategemma mix`.")
    else:
        if split not in get_dataset_split_names(data):
            return None
        dataset = load_dataset(data, split=split)
    return to_translation_rows(dataset)


def load_bouquet(split: str) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(BOUQUET, data_files=bouquet_files(split), split="train")
    return dataset.map(
        lambda row: {
            "id": row["uniq_id"],
            SOURCE_COLUMN: row["src_text"],
            TARGET_COLUMN: row["tgt_text"],
            "source_lang_code": BOUQUET_CODES[row["src_lang"]],
            "target_lang_code": BOUQUET_CODES[row["tgt_lang"]],
        },
        remove_columns=dataset.column_names,
    )


def load_moore_web(split: str) -> Any:
    """moore-web ``split`` (validation or test) in both directions."""
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset(MOORE_WEB, "mos-fra", split=split)
    dataset = dataset.filter(lambda row: bool(row["french"] and row["moore"]))
    directions = []
    for source, target, source_code, target_code in (
        ("french", "moore", FRENCH_CODE, MOORE_CODE),
        ("moore", "french", MOORE_CODE, FRENCH_CODE),
    ):
        directions.append(
            dataset.map(
                lambda row, s=source, t=target, sc=source_code, tc=target_code: {
                    SOURCE_COLUMN: row[s].strip(),
                    TARGET_COLUMN: row[t].strip(),
                    "source_lang_code": sc,
                    "target_lang_code": tc,
                },
                remove_columns=dataset.column_names,
            )
        )
    return concatenate_datasets(directions)


EVAL_SETS = {"bouquet": load_bouquet, "moore-web": load_moore_web}
DEFAULT_EVAL_SPLITS = {"bouquet": "dev", "moore-web": "test"}


def load_eval_set(name: str, split: str | None, direction: str) -> Any:
    """Load a named eval set, a mix directory, a data file or a Hub dataset, in ``direction``."""
    if name in EVAL_SETS:
        dataset = EVAL_SETS[name](split or DEFAULT_EVAL_SPLITS[name])
    else:
        dataset = load_translation_split(name, split or "test")
        if dataset is None:
            raise ValueError(f"{name} has no split {split or 'test'!r}.")
    if direction == "both":
        return dataset
    source_code, target_code = DIRECTIONS[direction]
    return dataset.filter(
        lambda row: (
            row["source_lang_code"] == source_code and row["target_lang_code"] == target_code
        )
    )
