"""Run rule-based complaint reason detection on negative IMDb reviews.

The script expects project-processed IMDb splits by default, where
``label == 0`` means negative. It can also handle the raw Kaggle/Maas-style
``sentiment`` column, where ``sentiment == 1`` means negative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.negative_reason_detector import NegativeReasonDetector  # noqa: E402


DEFAULT_INPUT_PATH = Path("data/processed/train.csv")
DEFAULT_OUTPUT_DIR = Path("reports/negative_reason_detection")
NESTED_EXPORT_COLUMNS = (
    "labels",
    "reason_scores",
    "heuristic_confidence",
    "supporting_sentences",
    "matched_patterns",
    "reasons",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect likely complaint reasons in negative IMDb reviews."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"CSV file to analyze. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/JSON exports. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--output-stem",
        default="negative_reason_detection",
        help="Base filename for exported CSV and JSON files.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the review text column.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Optional sentiment label column. If omitted, uses 'label' when "
            "available, then 'sentiment'."
        ),
    )
    parser.add_argument(
        "--negative-value",
        default=None,
        help=(
            "Optional value used to identify negative reviews. If omitted, "
            "uses 0 for 'label' and 1 for 'sentiment'."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of negative reviews to analyze.",
    )
    parser.add_argument(
        "--include-normalized-text",
        action="store_true",
        help="Include detector-normalized text in the exports.",
    )
    return parser.parse_args()


def load_imdb_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    return pd.read_csv(input_path)


def infer_label_column(df: pd.DataFrame, label_column: str | None) -> str:
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Missing label column: {label_column}")
        return label_column

    for candidate in ("label", "sentiment"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find a label column. Expected 'label' or 'sentiment', "
        "or pass --label-column."
    )


def infer_negative_value(label_column: str, negative_value: str | None) -> Any:
    if negative_value is not None:
        try:
            return int(negative_value)
        except ValueError:
            return negative_value

    if label_column == "label":
        return 0
    if label_column == "sentiment":
        return 1

    raise ValueError(
        "Please pass --negative-value for custom label columns "
        f"such as '{label_column}'."
    )


def filter_negative_reviews(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    negative_value: Any,
    limit: int | None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    negative_df = df[df[label_column] == negative_value].copy()
    negative_df = negative_df.reset_index(names="source_row")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer.")
        negative_df = negative_df.head(limit)

    if negative_df.empty:
        raise ValueError(
            f"No negative reviews found with {label_column} == {negative_value!r}."
        )

    negative_df[text_column] = negative_df[text_column].fillna("").astype(str)
    return negative_df


def detection_to_export_record(
    source_row: int,
    review_text: str,
    detection: dict[str, Any],
) -> dict[str, Any]:
    labels = detection["labels"]
    return {
        "source_row": source_row,
        "text": review_text,
        "labels": labels,
        "labels_pipe": "|".join(labels),
        "has_multiple_reasons": detection["has_multiple_reasons"],
        "sentence_count": detection["sentence_count"],
        "reason_scores": detection["reason_scores"],
        "heuristic_confidence": detection["heuristic_confidence"],
        "supporting_sentences": detection["supporting_sentences"],
        "matched_patterns": detection["matched_patterns"],
        "reasons": detection["reasons"],
        **(
            {"normalized_text": detection["normalized_text"]}
            if "normalized_text" in detection
            else {}
        ),
    }


def analyze_negative_reviews(
    negative_df: pd.DataFrame,
    text_column: str,
    include_normalized_text: bool,
) -> list[dict[str, Any]]:
    detector = NegativeReasonDetector()
    review_texts = negative_df[text_column].tolist()
    detections = detector.analyze_reviews(
        review_texts,
        include_normalized_text=include_normalized_text,
    )

    return [
        detection_to_export_record(
            source_row=int(source_row),
            review_text=review_text,
            detection=detection.to_dict(
                include_normalized_text=include_normalized_text
            ),
        )
        for source_row, review_text, detection in zip(
            negative_df["source_row"].tolist(),
            review_texts,
            detections,
            strict=True,
        )
    ]


def records_to_csv_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    csv_records = []
    for record in records:
        csv_record = record.copy()
        for column in NESTED_EXPORT_COLUMNS:
            csv_record[column] = json.dumps(csv_record[column], ensure_ascii=False)
        csv_records.append(csv_record)

    return pd.DataFrame(csv_records)


def save_results(
    records: list[dict[str, Any]],
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_stem}.csv"
    json_path = output_dir / f"{output_stem}.json"

    records_to_csv_frame(records).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return csv_path, json_path


def main() -> None:
    args = parse_args()

    df = load_imdb_data(args.input_path)
    label_column = infer_label_column(df, args.label_column)
    negative_value = infer_negative_value(label_column, args.negative_value)
    negative_df = filter_negative_reviews(
        df=df,
        text_column=args.text_column,
        label_column=label_column,
        negative_value=negative_value,
        limit=args.limit,
    )
    records = analyze_negative_reviews(
        negative_df=negative_df,
        text_column=args.text_column,
        include_normalized_text=args.include_normalized_text,
    )
    csv_path, json_path = save_results(
        records=records,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )

    print(f"Loaded {len(df)} reviews from {args.input_path}")
    print(
        f"Analyzed {len(records)} negative reviews "
        f"where {label_column} == {negative_value!r}"
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
