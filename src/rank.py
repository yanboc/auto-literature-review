import pandas as pd
import numpy as np
import faiss
from __future__ import annotations
from sentence_transformers import SentenceTransformer
import argparse
from pathlib import Path
from typing import Dict, Any

MODEL_NAME = "all-mpnet-base-v2"
ROOT_DIR = Path(__file__).parent.parent
SOURCE_PATH = ROOT_DIR / "configs" / "source_papers.csv"
TARGET_PATH = ROOT_DIR / "data" / "target_papers.csv"
SCORE_PATH = ROOT_DIR / "data" / "target_papers_score.csv"

BATCH_SIZE = 512
TOP_K = 500

def cal_embeddings(
    source_papers: pd.DataFrame, 
    target_papers: pd.DataFrame, 
    model: SentenceTransformer 
) -> Dict[str, Any]:
    """
    Calculate the similarity between source papers and target papers.

    args:
        method: the method to calculate the similarity, choose from "embedding" ...
    """
    source_titles = source_papers["title"].to_list()
    source_abstracts = source_papers["abstract"].to_list()
    source_info = [
        f"TITLE: {title}; ABS: {abstract}"
        for title, abstract in zip(source_titles, source_abstracts)
    ]

    target_titles = target_papers["title"].to_list()
    target_abstracts = target_papers["abstract"].to_list()
    target_info = [
        f"TITLE: {title}; ABS: {abstract}"
        for title, abstract in zip(target_titles, target_abstracts)
    ]

    source_embedding = model.encode(source_info, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    target_embedding = model.encode(target_info, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

    return source_embedding, target_embedding

def cal_similarity(
    source_embedding: "torch.Tensor",
    target_embedding: "torch.Tensor",
    method="faiss"
    ):
    if method == "faiss":
        target_embeddings = np.array(target_embeddings).astype('float32')
        source_embeddings = np.array(source_embeddings).astype('float32')

        index = faiss.IndexFlatIP(target_embedding.shape[1])
        index.add(target_embedding)

        print(f"FAISS index built: {index.ntotal} vectors")
    similarity = model.similarity(target_embedding, source_embedding)
    mean_similarity = similarity.mean(axis=1).tolist()
    return mean_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity Calculator")
    parser.add_argument(
        "--source", help="Path to source papers (.json)", default=SOURCE_PATH
    )
    parser.add_argument(
        "--target", help="Path to target papers (.csv)", default=TARGET_PATH
    )

    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    source_papers = pd.read_csv(args.source)
    target_papers = pd.read_csv(args.target)
    mean_similarity = cal_similarity(source_papers, target_papers, model)
    target_papers["mean_similarity"] = mean_similarity
    import pdb

    pdb.set_trace()
    target_papers.to_csv(SCORE_PATH, index=False)
