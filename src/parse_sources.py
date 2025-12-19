from pathlib import Path
from typing import Dict, Any
import time
import json
import logging
import requests
from lxml import etree as ET
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arxiv_id", type=str, required=False, default=None)
args = parser.parse_args()

ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ROOT_DIR = Path(__file__).parent.parent
SOURCE_PAPERS_PATH = ROOT_DIR / "configs" / "source_papers.csv"


def parse_from_arxiv(arxiv_id: str) -> Dict[str, Any]:
    """
    Parse paper information from arXiv API given an arXiv id.
    Output a dictionary of paper information with keys: 'arxiv_id' (str), 'title' (str), 'abstract' (str), and 'authors' (list of str).
    """
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            # parse XML
            root = ET.fromstring(response.content)
            # arXiv returns entry under 'entry' tag
            entry = root.find(f"{{{ATOM_NAMESPACE}}}entry")
            try:
                title = entry.find(f"{{{ATOM_NAMESPACE}}}title")
                title = title.text.strip() if title is not None else None

                abstract = entry.find(f"{{{ATOM_NAMESPACE}}}summary")
                if abstract is not None:
                    abstract = abstract.text.strip()
                else:
                    abstract = None

                authors = []
                for author in entry.findall(f"{{{ATOM_NAMESPACE}}}author"):
                    name = author.find(f"{{{ATOM_NAMESPACE}}}name")
                    if name is not None:
                        authors.append(name.text)

                return {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                }
            except Exception as e:
                raise Exception(
                    f"Parsing paper from arXiv API has successed.\
                    However, failed to parse paper information: {str(e)}"
                )
        else:
            raise Exception(
                f"Parsing paper from arXiv API has failed. \
                Status code: {response.status_code}. Response: {response.text}"
            )
    except Exception as e:
        raise Exception(
            f"Unexpected error when parsing paper from arXiv API. \
            Error message: {str(e)}"
        )


if __name__ == "__main__":
    source_papers = []

    if args.arxiv_id:
        paper = parse_from_arxiv(args.arxiv_id)
        source_papers.append(paper)
    else:
        papers = pd.read_csv(SOURCE_PAPERS_PATH, dtype={"arxiv_id": str})
        logging.info(f"Parsing {len(papers)} papers from arXiv API...")

        for index, paper in papers.iterrows():
            arxiv_id = paper["arxiv_id"]
            paper_info = parse_from_arxiv(arxiv_id)
            # 分别赋值每一列，避免类型不匹配问题
            papers.loc[index, 'title'] = paper_info["title"]
            papers.loc[index, 'authors'] = "; ".join(paper_info["authors"]) if paper_info["authors"] else ""
            papers.loc[index, 'abstract'] = paper_info["abstract"]
            time.sleep(1)  # arXiv policy

    papers.to_csv(SOURCE_PAPERS_PATH, index=False)
    logging.info(f"Parsed {len(source_papers)} papers from arXiv API.")
