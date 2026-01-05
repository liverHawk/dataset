from pathlib import Path

import logging
import argparse
import polars as pl

from lib.data import get_schema, prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    # 1. Delete unwanted columns
    logger.info("Deleting unwanted columns")
    delete_columns = [
        "id", "Flow ID", "ICMP Code", "ICMP Type", "Attempted Category"
    ]
    delete_columns = [col for col in delete_columns if col in df.columns]
    df = df.drop(delete_columns)

    df = prepare_data(df)

    return df


def save_data_info(df: pl.DataFrame, save_path: Path):
    logger.info("Saving column information")
    with open(save_path / "column_info.txt", "w") as f:
        f.write("Column Name,Data Type\n")
        for col, dtype in zip(df.columns, df.dtypes):
            dtype_str = str(dtype)
            f.write(f"{col},{dtype_str}\n")

    if "Label" not in df.columns:
        logger.warning("Label column not found. Adding dummy column")
        raise ValueError("Label column not found")
    logger.info("Saving label information")
    label_counts = df.group_by("Label").agg(pl.len().alias("count")).sort("count", descending=True)

    with open(save_path / "label_info.txt", "w") as f:
        f.write("Label,Count\n")
        for label, count in zip(label_counts["Label"], label_counts["count"]):
            f.write(f"{label},{count}\n")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="CICIDS2017_improved",
        help="input Dataset name"
    )
    return parser.parse_args()


def main():
    args = load_args()
    path = Path(f'../{args.dataset}')
    files = list(path.glob('*.csv'))

    unified_schema = get_schema(files)

    for file in files:
        df = pl.read_csv(file, schema=unified_schema)
        try:
            df = clean_data(df)
        except Exception as e:
            logger.error(f"Error cleaning data for {file}: {e}")
            continue

        file_name = file.stem + "_cleaned.csv"
        save_base_path = Path('./cleaned') / file.parent.name
        save_base_path.mkdir(parents=True, exist_ok=True)
        df.write_csv(save_base_path / file_name)

    df = pl.read_csv(files[0], schema=unified_schema)
    save_data_info(df, path)


if __name__ == "__main__":
    main()
