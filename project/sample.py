import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import polars as pl
    from pathlib import Path

    DATASET = "CICDDoS2019"
    path = Path(f"../{DATASET}")

    files = list(path.glob("*.csv"))

    dfs = []
    for file in mo.status.progress_bar(files, title="Loading", show_eta=True, show_rate=True):
        # df = pl.read_csv(file, columns=["SimillarHTTP"], infer_schema_length=0)
        df = pl.read_csv(file, columns=["SimillarHTTP"], schema_overrides={ "SimillarHTTP": pl.Utf8 })
        dfs.append(df)

    combined_df = pl.concat(dfs)
    return (combined_df,)


@app.cell
def _(combined_df):
    unique_list = combined_df["SimillarHTTP"].unique()
    for idx in range(len(unique_list)):
        print(unique_list[idx])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
