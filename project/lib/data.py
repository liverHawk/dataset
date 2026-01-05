import polars as pl
import logging

logger = logging.getLogger(__name__)


def get_schema(files):
    unified_schema = {}
    for file in files:
        current_schema = pl.read_csv(file, n_rows=0).schema
        for col, dtype in current_schema.items():
            if col not in unified_schema:
                unified_schema[col] = dtype
            else:
                if unified_schema[col] != dtype:
                    if dtype.is_float() or unified_schema[col].is_float():
                        unified_schema[col] = pl.Float64

    return unified_schema


def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    # 2. replace int and -inf to nan
    logger.info("Replacing int and -inf to nan")
    numeric_columns = [
        col for col, dtype in zip(df.columns, df.dtypes) if dtype.is_numeric()
    ]
    for col in numeric_columns:
        df = df.with_columns(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    df = df.drop_nulls()

    # 3. drop rows which have negative values
    logger.info("Dropping rows which have negative values")
    for col in numeric_columns:
        if col in df.columns:
            df = df.filter(pl.col(col) >= 0)
    return df