"""
Fetch data from Huggingface
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from datasets import load_dataset


def fetch_dataset_data(
    dataset_info: str,
    target_papers: pd.DataFrame,
    to_csv: bool = True,
    csv_path: Optional[Path] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Fetch data from Huggingface dataset.
    Clean these datasets and get one list of target papers.

    Args:
        dataset_info: a row of dataset info in the csv file
        to_json: whether to save the data to a json file
        json_path: the path to save the json file

    Returns:
        data_dicts: a list of data dictionaries
    """

    dataset_name = dataset_info["hf_name"]
    conf_name = dataset_info["conference"]
    conf_year = dataset_info["year"]

    if not csv_path:
        root_dir = Path(__file__).parent.parent
        csv_path = Path(root_dir / "data" / f"{conf_name}-{conf_year}.csv")

    if not force_reload and csv_path.exists():
        print(f"Loading data from {csv_path}. Not fetching from remote.")
        df = pd.read_csv(csv_path)
        to_csv = False

    elif force_reload and csv_path.exists():
        print(
            f"Data already exists in {csv_path}. Still reloading since force_reload is set to {force_reload}."
        )
        os.remove(csv_path)
        to_csv = True

    dataset = load_dataset(dataset_name)["train"]
    df = dataset.to_pandas()

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

    current_conf_df = df[["title", "authors", "abstract", "conf_info"]].reset_index(
        drop=True
    )
    target_papers = pd.concat([target_papers, current_conf_df], ignore_index=True)

    return target_papers


if __name__ == "__main__":
    print("Fetching dataset data...")
    root_dir = Path(__file__).parent.parent
    dataset_infos = pd.read_csv(root_dir / "assets" / "datasets.csv")
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
        target_papers = fetch_dataset_data(dataset_info, target_papers=target_papers)

    target_papers.to_csv(root_dir / "assets" / "target_papers.csv", index=False)
