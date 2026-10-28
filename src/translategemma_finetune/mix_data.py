"""Build French <-> Mooré training, validation and test JSONL.

Ported from the mos-fra part of the Hy-MT2 mixer (Hy-MT2-Finetune/scripts/mix_translation_data.py):
each source gets a fixed share of the mix, and each sampled row is assigned one direction,
biased toward generating Mooré. Sentences from every evaluation set (moore-web validation/test,
FLORES+ dev/devtest, BOUQuET dev/test) are removed from training.

Rows have ``source_text``, ``target_text``, ``source_lang_code``, ``target_lang_code`` and
``origin`` (the source name), so training reads the direction from each row.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translategemma_finetune.data import (
    BOUQUET,
    DEFAULT_MIX_DIR,
    FRENCH_CODE,
    MOORE_CODE,
    MOORE_WEB,
    bouquet_files,
)

NLLB_BURKINA = "burkimbia/nllb-burkina-parallel"
FLORES = "openlanguagedata/flores_plus"
# Starting thresholds from the Hy-MT2 mix, chosen from the score distributions of each dataset.
MOORE_WEB_MIN_LASER = 0.70
MOORE_WEB_MIN_COMET_QE = 0.52
NLLB_MIN_SCORE = 1.06
NLLB_MIN_LID_PROB = 0.9
NLLB_MIN_LENGTH_RATIO = 0.5


def score_at_least(row: dict[str, Any], column: str, minimum: float) -> bool:
    value = row.get(column)
    return value is not None and value >= minimum


def score_missing_or_at_least(row: dict[str, Any], column: str, minimum: float) -> bool:
    value = row.get(column)
    return value is None or value >= minimum


def moore_web_reviewed(row: dict[str, Any]) -> bool:
    return bool(row["reviewed"])


def moore_web_aligned(row: dict[str, Any]) -> bool:
    # `news` is excluded: its Mooré side is a summary-style rewrite, not a translation.
    return (
        not row["reviewed"]
        and row["source"] in {"conseils", "lexicon", "digital-defs"}
        and score_at_least(row, "laser_score", MOORE_WEB_MIN_LASER)
        and score_missing_or_at_least(row, "comet_qe", MOORE_WEB_MIN_COMET_QE)
    )


def moore_web_lexicon(row: dict[str, Any]) -> bool:
    return row["source"] in {"lexicon_entries", "digital-terms"}


def nllb_fra_mos(row: dict[str, Any]) -> bool:
    return (
        row["score"] >= NLLB_MIN_SCORE
        and row["length_ratio"] >= NLLB_MIN_LENGTH_RATIO
        and row["src_lid"] == "fra_Latn"
        and row["trg_lid"] == "mos_Latn"
        and row["src_lid_prob"] >= NLLB_MIN_LID_PROB
        and row["trg_lid_prob"] >= NLLB_MIN_LID_PROB
    )


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    moore_column: str
    french_column: str
    weight: float
    keep: Callable[[dict[str, Any]], bool]
    pivot_column: str | None = None


# Order matters: when deduplicating, the earlier source keeps the row.
SOURCES = [
    SourceSpec(
        "moore_web_reviewed", MOORE_WEB, "mos-fra", "moore", "french", 0.30, moore_web_reviewed
    ),
    SourceSpec(
        "moore_web_aligned", MOORE_WEB, "mos-fra", "moore", "french", 0.35, moore_web_aligned
    ),
    SourceSpec(
        "nllb_fra_mos_pivot",
        NLLB_BURKINA,
        "fra-mos",
        "trg_text",
        "src_text",
        0.25,
        nllb_fra_mos,
        pivot_column="pivot_text",
    ),
    # Dictionary rows stay at 10%: too many single-word pairs teach short outputs.
    SourceSpec(
        "moore_web_lexicon", MOORE_WEB, "mos-fra", "moore", "french", 0.10, moore_web_lexicon
    ),
]


@dataclass(frozen=True)
class PairExample:
    origin: str
    moore: str
    french: str


def read_hf_dataset(
    name: str, split: str = "train", config: str | None = None
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(name, config, split=split)]


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_text(text: str) -> str:
    """Matching key for dedupe and decontamination: case, punctuation and spacing are ignored."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def held_out_texts() -> set[str]:
    """Normalized sentences from every evaluation set, so they never reach training data."""
    from datasets import load_dataset

    texts = set()
    for split in ("validation", "test"):
        for row in read_hf_dataset(MOORE_WEB, split=split, config="mos-fra"):
            texts.update(
                normalize_text(row[column]) for column in ("moore", "french") if row[column]
            )
    for split in ("dev", "devtest"):
        for lang in ("mos_Latn", "fra_Latn"):
            texts.update(
                normalize_text(row["text"]) for row in read_hf_dataset(FLORES, split, lang)
            )
    for row in load_dataset(
        BOUQUET, data_files=bouquet_files("dev") + bouquet_files("test"), split="train"
    ):
        texts.update(normalize_text(row[column]) for column in ("src_text", "tgt_text"))
    texts.discard("")
    return texts


def load_source(
    spec: SourceSpec,
    rows: list[dict[str, Any]],
    held_out: set[str],
    seen: set[tuple[str, str]],
) -> tuple[list[PairExample], Counter]:
    stats = Counter()
    examples = []
    for row in rows:
        if not spec.keep(row):
            continue
        stats["kept_by_filter"] += 1
        moore = clean_text(row.get(spec.moore_column))
        french = clean_text(row.get(spec.french_column))
        if not moore or not french:
            stats["empty"] += 1
            continue

        texts = [moore, french]
        if spec.pivot_column and row.get(spec.pivot_column):
            texts.append(row[spec.pivot_column])
        if any(normalize_text(text) in held_out for text in texts):
            stats["held_out"] += 1
            continue

        key = (normalize_text(moore), normalize_text(french))
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        examples.append(PairExample(spec.name, moore, french))
    stats["usable"] = len(examples)
    return examples, stats


def load_sources() -> dict[str, list[PairExample]]:
    held_out = held_out_texts()
    print(f"Held-out sentences (moore-web validation/test, FLORES+, BOUQuET): {len(held_out)}")

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    pools = {}
    for spec in SOURCES:
        key = (spec.dataset, spec.config)
        if key not in cache:
            cache[key] = read_hf_dataset(spec.dataset, config=spec.config)
        pools[spec.name], stats = load_source(spec, cache[key], held_out, seen)
        print(
            f"{spec.name}: {stats['usable']} usable "
            f"(filter kept {stats['kept_by_filter']}, held-out {stats['held_out']}, "
            f"duplicates {stats['duplicate']}, empty {stats['empty']})"
        )
    return pools


def allocation(weights: dict[str, float], total: int) -> dict[str, int]:
    weight_sum = sum(weights.values())
    raw_counts = {key: (weight / weight_sum) * total for key, weight in weights.items()}
    counts = {key: int(value) for key, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw_counts, key=lambda key: raw_counts[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def sample_pairs(rng: random.Random, pairs: list[PairExample], count: int) -> list[PairExample]:
    if count <= len(pairs):
        return rng.sample(pairs, count)
    sampled = list(pairs)
    sampled.extend(rng.choice(pairs) for _ in range(count - len(pairs)))
    rng.shuffle(sampled)
    return sampled


def translation_row(example: PairExample, into_moore: bool) -> dict[str, str]:
    if into_moore:
        return {
            "source_text": example.french,
            "target_text": example.moore,
            "source_lang_code": FRENCH_CODE,
            "target_lang_code": MOORE_CODE,
            "origin": example.origin,
        }
    return {
        "source_text": example.moore,
        "target_text": example.french,
        "source_lang_code": MOORE_CODE,
        "target_lang_code": FRENCH_CODE,
        "origin": example.origin,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_train(args: argparse.Namespace, rng: random.Random) -> list[dict[str, str]]:
    pools = load_sources()
    counts = allocation(
        {spec.name: spec.weight for spec in SOURCES if pools[spec.name]}, args.total_examples
    )
    rows = []
    print("\nSampled rows:")
    for name, count in counts.items():
        pool = pools[name]
        print(f"  {name}: {count} from {len(pool)} ({count / len(pool):.2f}x)")
        rows.extend(
            translation_row(example, into_moore=rng.random() < args.into_moore_ratio)
            for example in sample_pairs(rng, pool, count)
        )
    rng.shuffle(rows)
    return rows


def build_eval(
    split: str, max_per_direction: int | None, rng: random.Random
) -> list[dict[str, str]]:
    """moore-web ``split`` in both directions, optionally sampled per direction."""
    pairs = []
    for row in read_hf_dataset(MOORE_WEB, split=split, config="mos-fra"):
        moore, french = clean_text(row["moore"]), clean_text(row["french"])
        if moore and french:
            pairs.append(PairExample(f"moore_web_{split}", moore, french))
    if max_per_direction is not None and max_per_direction < len(pairs):
        pairs = rng.sample(pairs, max_per_direction)
    rows = [translation_row(pair, into_moore=True) for pair in pairs]
    rows += [translation_row(pair, into_moore=False) for pair in pairs]
    return rows


def direction_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter(f"{row['source_lang_code']}->{row['target_lang_code']}" for row in rows)
    return ", ".join(f"{direction}: {count}" for direction, count in sorted(counts.items()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="translategemma mix", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_MIX_DIR))
    parser.add_argument("--total-examples", type=int, default=50_000)
    parser.add_argument(
        "--into-moore-ratio",
        type=float,
        default=0.6,
        help="Share of training rows translated into Mooré; the rest go into French.",
    )
    parser.add_argument(
        "--validation-rows-per-direction",
        type=int,
        default=500,
        help="moore-web validation rows per direction, for evaluation during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args(argv)
    rng = random.Random(args.seed)
    outputs = {
        "train": build_train(args, rng),
        "validation": build_eval("validation", args.validation_rows_per_direction, rng),
        "test": build_eval("test", None, rng),
    }
    print()
    for split, rows in outputs.items():
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        print(f"Wrote {len(rows)} rows to {path} ({direction_counts(rows)})")


if __name__ == "__main__":
    main()
