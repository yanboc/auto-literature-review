"""
Fetch data from Huggingface
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import load_dataset

def fetch_dataset_data(
    dataset_info: str,
    to_json: bool = True,
    json_path: Optional[Path] = None,
    force_reload: bool = False,
    ) -> Dict[str, Any]:
    """
    Fetch data from Huggingface dataset
    
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

    if not json_path:
        root_dir = Path(__file__).parent.parent
        json_path = Path(root_dir / "data" / f"{conf_name}-{conf_year}.json")

    if not force_reload and json_path.exists():
        print(f"Loading data from {json_path}. Not fetching from Huggingface.")
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        return data_dict
    elif force_reload and json_path.exists():
        print(f"Data already exists in {json_path}. Still reloading since force_reload is set to True.")
        os.remove(json_path)
    dataset = load_dataset(dataset_name)["train"]
    data_dict = dataset.to_dict()


    if to_json:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise Exception(f"Failed to save JSON file: {str(e)}")

    return data_dict

if __name__ == "__main__":
    print("Fetching dataset data...")
    root_dir = Path(__file__).parent.parent
    dataset_infos = pd.read_csv(root_dir / "assets" / "datasets.csv")
    dataset_infos.columns = dataset_infos.columns.str.strip()
    for _, dataset_info in dataset_infos.iterrows():
        fetch_dataset_data(dataset_info, to_json=True, json_path=None)