from pathlib import Path

import logging
import argparse
import polars as pl
import coloredlogs
import json
import os

coloredlogs.install(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sample_by_label(
    lf: pl.LazyFrame,
    label_column: str = "Label",
    n_samples_per_label: int = None,
    fraction_per_label: float = None,
    seed: int = 42
) -> dict[str, pl.DataFrame]:
    """
    ラベルごとにデータをランダムサンプリングする
    
    Args:
        lf: サンプリング対象のLazyFrame
        label_column: ラベルカラム名
        n_samples_per_label: 各ラベルから取得するサンプル数（指定する場合はfraction_per_labelと同時に指定不可）
        fraction_per_label: 各ラベルから取得する割合（0.0-1.0、指定する場合はn_samples_per_labelと同時に指定不可）
        seed: ランダムシード
    
    Returns:
        ラベル名をキー、サンプリングされたDataFrameを値とする辞書
    """
    if n_samples_per_label is not None and fraction_per_label is not None:
        raise ValueError("Cannot specify both n_samples_per_label and fraction_per_label")
    
    if n_samples_per_label is None and fraction_per_label is None:
        raise ValueError("Must specify either n_samples_per_label or fraction_per_label")
    
    logger.info(f"Sampling data by label column: {label_column}")
    
    # ラベルごとのデータ数を確認（遅延評価）
    label_counts_lf = lf.group_by(label_column).agg(pl.len().alias("count")).sort("count", descending=True)
    label_counts = label_counts_lf.collect()
    
    logger.info("Label distribution:")
    for row in label_counts.iter_rows(named=True):
        logger.info(f"  {row[label_column]}: {row['count']} samples")
    
    # ラベルごとにサンプリング
    sampled_dfs = {}
    for label_row in label_counts.iter_rows(named=True):
        label_value = label_row[label_column]
        label_count = label_row['count']
        
        # ラベルごとのデータを取得（遅延評価）
        label_lf = lf.filter(pl.col(label_column) == label_value)
        
        # サンプリング数を決定
        if n_samples_per_label is not None:
            sample_size = min(n_samples_per_label, label_count)
        else:  # fraction_per_label
            sample_size = max(1, int(label_count * fraction_per_label))
        
        logger.info(f"  Sampling {sample_size} samples from '{label_value}' (total: {label_count})")
        
        # ラベルごとのデータをDataFrameに変換してからサンプリング
        label_df = label_lf.collect()
        # ランダムサンプリング
        sampled_label_df = label_df.sample(n=sample_size, seed=seed)
        sampled_dfs[label_value] = sampled_label_df
    
    # サンプリング後のラベル分布を確認
    logger.info("Sampled label distribution:")
    for label_value, sampled_df in sampled_dfs.items():
        logger.info(f"  {label_value}: {len(sampled_df)} samples")
    
    return sampled_dfs

def get_type(dtype: str) -> pl.DataType:
    if dtype == "String":
        return pl.Utf8
    elif dtype == "Int64":
        return pl.Int64
    elif dtype == "Float64":
        return pl.Float64
    else:
        raise ValueError(f"Unknown data type: {dtype}")

def load_cleaned_data(dataset_path: Path, count_rows: bool = False) -> pl.LazyFrame:
    """
    cleanedディレクトリからデータを読み込む（遅延評価）
    
    Args:
        dataset_path: cleanedデータセットのパス
        count_rows: 総行数をカウントするかどうか
    
    Returns:
        読み込んだLazyFrame
    """
    csv_files = list(dataset_path.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.endswith("_info.txt")]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {dataset_path}")
    
    logger.info(f"Scanning {len(csv_files)} CSV files from {dataset_path}")
    dataset_name = dataset_path.name
    if os.path.exists(f"metadata/{dataset_name}.json"):
        with open(f"metadata/{dataset_name}.json", "r") as f:
            schema = json.load(f)
        if "schema" in schema:
            schema = schema["schema"]
        else:
            schema = {}
    
    schema = {k: get_type(v) for k, v in schema.items()}
    
    # すべてのCSVファイルをscan_csvで読み込む（遅延評価）
    lazy_frames = []
    for csv_file in csv_files:
        logger.info(f"  Scanning {csv_file.name}")
        lf = pl.scan_csv(csv_file, schema_overrides=schema)
        lazy_frames.append(lf)
    
    # すべてのLazyFrameを結合（遅延評価のまま）
    combined_lf = pl.concat(lazy_frames)
    
    # 行数を取得するために一度collect（オプション、ログ出力用）
    # 大規模ファイルの場合はコストが高いので、必要に応じてコメントアウト
    if count_rows:
        logger.info("Counting total rows...")
        total_rows = combined_lf.select(pl.len()).collect().item()
        logger.info(f"Total rows: {total_rows}")
    
    return combined_lf


def load_args():
    parser = argparse.ArgumentParser(description="ラベルごとにデータをランダムサンプリング")
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="CICIDS2017_improved",
        help="データセット名（cleanedディレクトリ内のサブディレクトリ名）"
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int, default=None,
        help="各ラベルから取得するサンプル数"
    )
    parser.add_argument(
        "-f", "--fraction",
        type=float, default=None,
        help="各ラベルから取得する割合（0.0-1.0）"
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="ランダムシード"
    )
    parser.add_argument(
        "--label-column",
        type=str, default="Label",
        help="ラベルカラム名"
    )
    parser.add_argument(
        "--count-rows",
        action="store_true",
        help="総行数をカウントする（大規模ファイルでは時間がかかる場合があります）"
    )

    return parser.parse_args()


def main():
    args = load_args()
    
    # データセットパス
    dataset_path = Path('./cleaned') / args.dataset
    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    # データを読み込む（遅延評価）
    lf = load_cleaned_data(dataset_path, count_rows=args.count_rows)
    if args.count_rows:
        exit()
    
    # ラベルごとにサンプリング
    sampled_dfs = sample_by_label(
        lf,
        label_column=args.label_column,
        n_samples_per_label=args.n_samples,
        fraction_per_label=args.fraction,
        seed=args.seed
    )
    
    # 出力ディレクトリを決定
    output_dir = Path('./sampled') / f"n_{args.n_samples}_s_{args.seed}" / args.dataset
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ラベルごとにファイルを保存
    total_sampled = 0
    for label_value, sampled_df in sampled_dfs.items():
        # ファイル名に使えない文字を置換
        safe_label = str(label_value).replace(" - ", "-").replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
        output_file = output_dir / f"{safe_label}_sampled.csv"
        
        logger.info(f"Saving '{label_value}' samples to {output_file} ({len(sampled_df)} rows)")
        sampled_df.write_csv(output_file)
        total_sampled += len(sampled_df)
    
    logger.info(f"Done! Sampled {total_sampled} rows across {len(sampled_dfs)} labels")


if __name__ == "__main__":
    main()
