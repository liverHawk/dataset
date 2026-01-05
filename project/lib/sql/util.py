import polars as pl


def escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def safe_convert_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    df_clean = df.clone()

    numeric_cols = [
        col for col in df_clean.columns if df_clean[col].dtype.is_numeric()
    ]
    for col in numeric_cols:
        df_clean = df_clean.with_columns(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df_clean
