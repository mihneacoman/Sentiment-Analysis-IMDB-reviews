from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/imdb_maas_2011/train.csv")
PROCESSED_DATA_DIR = Path("data/processed")

RANDOM_STATE = 42

TRAIN_SIZE_PER_CLASS = 4000
VALID_SIZE_PER_CLASS = 1000
TEST_SIZE_PER_CLASS = 1000


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Expected columns: 'text' and 'sentiment'.")

    df = df[["text", "sentiment"]]

    # Raw dataset convention: 0 = positive, 1 = negative
    # Project convention: 0 = negative, 1 = positive
    df["label"] = df["sentiment"].map({
        0: 1,
        1: 0,
    })

    df = df[["text", "label"]]

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"Unexpected labels: {df['label'].unique()}")

    return df


def sample_balanced_data(df: pd.DataFrame, samples_per_class: int) -> pd.DataFrame:
    negative_sample = df[df["label"] == 0].sample(
        n=samples_per_class,
        random_state=RANDOM_STATE,
    )

    positive_sample = df[df["label"] == 1].sample(
        n=samples_per_class,
        random_state=RANDOM_STATE,
    )

    sampled_df = pd.concat([negative_sample, positive_sample], axis=0)

    return sampled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_per_class = TRAIN_SIZE_PER_CLASS + VALID_SIZE_PER_CLASS + TEST_SIZE_PER_CLASS
    sampled_df = sample_balanced_data(df, total_per_class)

    train_df, temp_df = train_test_split(
        sampled_df,
        train_size=TRAIN_SIZE_PER_CLASS * 2,
        stratify=sampled_df["label"],
        random_state=RANDOM_STATE,
    )

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=VALID_SIZE_PER_CLASS * 2,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split(df: pd.DataFrame, split_name: str) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"{split_name}.csv"
    df.to_csv(output_path, index=False)

    print(
        f"Saved {output_path}: "
        f"{len(df)} rows, labels={df['label'].value_counts().to_dict()}"
    )


def main() -> None:
    raw_df = load_raw_data()
    df = normalize_columns(raw_df)

    train_df, valid_df, test_df = create_splits(df)

    save_split(train_df, "train")
    save_split(valid_df, "valid")
    save_split(test_df, "test")


if __name__ == "__main__":
    main()