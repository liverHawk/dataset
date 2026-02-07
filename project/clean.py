from pathlib import Path

import logging
import argparse
import polars as pl
import os
import json
import coloredlogs

from lib.data import get_schema, prepare_data

coloredlogs.install(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    # 1. strip whitespace from all columns
    logger.info("Stripping whitespace from all columns")
    df = df.rename({ col: col.strip() for col in df.columns })

    # 2. Delete unwanted columns
    logger.info("Deleting unwanted columns")
    delete_columns = [
        "id", "Flow ID", "ICMP Code", "ICMP Type", "Attempted Category",
        "Unnamed: 0", ""
    ]
    delete_columns = [col for col in delete_columns if col in df.columns]
    df = df.drop(delete_columns)

    df = prepare_data(df)

    return df


def save_data_info(files: list[Path], save_path: Path):
    logger.info("Saving column information")
    unified_schema = get_schema(files)
    df = pl.read_csv(files[0], schema=unified_schema)
    with open(save_path / "column_info.txt", "w") as f:
        f.write("Column Name,Data Type\n")
        for col, dtype in zip(df.columns, df.dtypes):
            dtype_str = str(dtype)
            f.write(f"{col},{dtype_str}\n")

    if "Label" not in df.columns:
        logger.warning("Label column not found. Adding dummy column")
        return
    dfs = []
    for file in files:
        df = pl.read_csv(file, columns=["Label"])
        dfs.append(df)
    df = pl.concat(dfs)
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

        if os.path.exists(f"metadata/{args.dataset}.json"):
            with open(f"metadata/{args.dataset}.json", "r") as f:
                metadata = json.load(f)
            if metadata.get("column_mapping"):
                mapping = metadata["column_mapping"]
                logger.info(f"Renaming columns according to mapping")
                df = df.rename(mapping)

        file_name = file.stem + "_cleaned.csv"
        save_base_path = Path('./cleaned') / file.parent.name
        save_base_path.mkdir(parents=True, exist_ok=True)
        df.write_csv(save_base_path / file_name)
    

    
    logger.info("Saving data info for original dataset")
    save_data_info(files, path)

    logger.info("Saving data info for cleaned dataset")
    new_path = Path('./cleaned') / args.dataset
    files = list(new_path.glob('*.csv'))
    save_data_info(files, new_path)
    logger.info("Done")

if __name__ == "__main__":
    main()
