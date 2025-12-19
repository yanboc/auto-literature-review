# Auto Literature Review Assistant

![Language: Python](https://img.shields.io/badge/Language-Python-blue?logo=python) ![Subject: CS/ML/AI](https://img.shields.io/badge/Subject-CS/ML/AI-yellowgreen) ![Model: MPNet+Qwen-3](https://img.shields.io/badge/Model-MPNet/Qwen--3-ff69b4) 

👉 [中文版简介](./README_CH.md)

Hello 🤗! We've developed an **Auto Literature Review Assistant** with integrated AI capabilities that can automatically generate a literature list based on user-provided source references and automatically organize & save AI-generated literature summaries. More importantly, since the entire project's codebase is very concise (~200 lines), you can easily add the features you need.

The main features of the current version include:

1. **（Target Literature Database）** We maintain a list of *accepted papers from past AI conferences* obtained from [Hugging Face](https://huggingface.co/) (see `assets/datasets.csv`). Based on this list, we clean and organize the metadata of target papers (including titles, authors, abstracts, PDFs, keywords, etc.) into a **target literature database**.
2. **（Parse Source Literature）** Users need to provide the arXiv id of source papers. The system can automatically retrieve the metadata of source papers based on arXiv id.
3. **（Initial Screening Based on Title and Abstract）** We estimate the similarity between source papers and target papers based on [Sentence Transformers](https://huggingface.co/sentence-transformers).
4. **（AI/Agentic Features - To Be Implemented）** For articles after initial screening, we will use AI capabilities to summarize PDF full texts (output in `markdown` format) and perform further comparison on the summarized articles, thereby generating a more precise list of relevant literature.
5. **（Query Records - To Be Implemented）** After a complete literature query, the system will store necessary information for subsequent viewing or further analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1. Download Data from 🤗

The dataset information containing paper metadata is manually saved in `assets/datasets.csv`. To expand the search scope, please add entries to this file as needed. After running

```bash
python3 src/fetch.py
```

data files in the format `{conf_name}-{conf_year}.csv` will be generated in the `data` folder, along with a summary file named `target_papers.csv`, which contains the titles, abstracts, and author information of all papers.

### Step 2. Parse Source Paper Information

Subsequent similarity calculations require metadata of source papers (including titles, authors, abstracts, etc.). You can automatically parse paper metadata by providing the *arXiv* id of papers:

```bash
python src/parse_sources.py
```

Alternatively, you can manually fill in the information following the example template in `configs/example_source_papers.csv` (rename the file to `configs/source_papers.csv` after filling it in).

### Step 3. Initial Screening Based on Title and Abstract

Run the following command to perform similarity calculation and ranking:

```bash
python src/rank.py
```
