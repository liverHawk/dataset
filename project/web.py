from pathlib import Path
import streamlit as st
import duckdb
import polars as pl

from lib.data import get_schema, prepare_data

st.title("CSV Viewer with DuckDB")

conn = duckdb.connect()


def cache_csv_to_parquet(csv_file, parquet_file, status: st.empty) -> bool:
    if not parquet_file.exists():
        status.write("Converting CSV to Parquet")

        try:
            df = pl.read_csv(csv_file)
            df = prepare_data(df)
            df.write_parquet(parquet_file)
            return True
        except Exception as e:
            status.write(f"Error converting CSV to Parquet: {e}")
            return False
    return False


csv_dir = st.text_input(
    "CSV Directory",
    value="../",
    placeholder="Enter the directory containing CSV files",
    help="Enter the directory containing CSV files"
)

csv_path = Path(csv_dir).expanduser().resolve() if csv_dir else None

if not csv_dir:
    st.info("Please enter a valid directory path")
    st.stop()
elif csv_path is None or not csv_path.exists():
    st.warning(f"Directory {csv_dir} does not exist")
    st.stop()
else:
    try:
        sub_dirs = sorted([p for p in csv_path.iterdir() if p.is_dir()])
    except Exception as e:
        st.warning(f"Error reading directory {csv_dir}: {e}")
        st.stop()

    # selected_dataset_path
