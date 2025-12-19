"""
Fetch data from Huggingface
"""

import os
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset


ROOT_DIR = Path(__file__).parent.parent


def fetch_dataset_data(
    dataset_info: Dict[str, Any], to_csv: bool = True, csv_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch data from Huggingface dataset or load in existing files.
    Clean these data and get a collection of target papers (in CSV).

    Args:
        dataset_info: a row of dataset info in the csv file, in Dict[str].
        to_csv: whether to save the data
        csv_path: the path to save the csv file

    Returns:
        current_conf_papers: clean form of target papers
    """

    dataset_name = dataset_info["hf_name"]
    conf_name = dataset_info["conf_name"]
    conf_year = dataset_info["conf_year"]

    if csv_path is None:
        csv_path = Path(ROOT_DIR / "data" / f"{conf_name}-{conf_year}.csv")

    if csv_path.exists():
        print(f"Loading data from {csv_path}. Not fetching from remote.")
        df = pd.read_csv(csv_path)
        to_csv = False

    # Fetch data from Hugging Face
    dataset = load_dataset(dataset_name)["train"]
    df = dataset.to_pandas()

    # Save the fetched data to CSV
    if to_csv:
        df.to_csv(csv_path, index=False)

    # Data processing
    print(f"Processing data from {dataset_name}...")
    df = df.drop(columns="embedding", errors="ignore")
    df["conf_info"] = f"{conf_name}-{conf_year}"
    if conf_name == "icml":
        df = df.rename(columns={"Download PDF": "pdf"})
    elif conf_name == "neurips":
        df = df.rename(columns={"Paper": "pdf"})

    # Only preserve title, authors, abstract, and conf_info columns
    current_conf_papers = df[["title", "authors", "abstract", "conf_info"]].reset_index(
        drop=True
    )
    return current_conf_papers


if __name__ == "__main__":

    print("Fetching dataset data...")
    dataset_infos = pd.read_csv(ROOT_DIR / "assets" / "datasets.csv")
    dataset_infos.columns = dataset_infos.columns.str.strip()
    target_papers = pd.DataFrame(
        {
            "title": [],
            "authors": [],
            "abstract": [],
            "conf_info": [],
        }
    ).reset_index(drop=True)
    for _, dataset_info in dataset_infos.iterrows():
        current_conf_papers = fetch_dataset_data(dataset_info.to_dict())
        target_papers = pd.concat(
            [target_papers, current_conf_papers], ignore_index=True
        )

    target_papers.to_csv(ROOT_DIR / "assets" / "target_papers.csv", index=False)
