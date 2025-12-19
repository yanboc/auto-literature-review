# 自动文献综述助手

![Language: Python](https://img.shields.io/badge/Language-Python-blue?logo=python) ![Subject: CS/ML/AI](https://img.shields.io/badge/Subject-CS/ML/AI-yellowgreen) ![Model: MPNet+Qwen-3](https://img.shields.io/badge/Model-MPNet/Qwen--3-ff69b4)

大家好🤗！我们开发了一个集成 AI 功能的**自动文献综述助手**，能够根据用户提供的源参考文献自动生成文献清单，并自动整理与保存 AI 生成的文献总结。更重要的是，由于整个项目的代码非常简洁（约 200 行），你可以轻松地添加所需的功能。

当前版本的主要功能包括：

1. **（获取目标文献库）** 我们维护了一份从 [Hugging Face](https://huggingface.co/) 获取的*历届 AI 会议接收论文清单*（见 `assets/datasets.csv`）。基于这份清单，我们将目标论文的元信息（包括题目、作者、摘要、PDF、关键词等）清洗并整理成一个**目标文献库**。
2. **（解析源文献）** 用户需要提供源论文的 arXiv id。系统可以根据 arXiv id 自动获取源论文的元信息。
3. **（基于标题与摘要的初筛）** 我们基于 [Sentence Transformers](https://huggingface.co/sentence-transformers) 估计源论文与目标论文的相似度。
4. **（AI/Agentic 功能 - 待实现）** 对于初筛后的文章，我们将使用 AI 功能对 PDF 全文进行总结（输出为 `markdown` 格式），并对总结后的文章进行进一步比对，从而生成一份更精准的相关文献清单。
5. **（查询记录 - 待实现）** 在一次完整的文献查询后，系统会将必要的信息存储，以便后续查看或在此基础上继续分析。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### Step 1. 从🤗下载数据

论文元数据所在的数据集信息已手动保存在 `assets/datasets.csv` 中。如需扩大检索范围，请自行在该文件中添加条目。运行

```bash
python3 src/fetch.py
```

后，在 `data` 文件夹中会生成类似 `{conf_name}-{conf_year}.csv` 格式的数据文件，以及一个名为 `target_papers.csv` 的汇总文件，其中包含所有论文的标题、摘要与作者信息。

### Step 2. 解析源论文信息

后续相似度计算需要源论文的元信息（包括标题、作者、摘要等）。你可以通过提供论文的 *arXiv* id 来自动解析论文元信息：

```bash
python src/parse_sources.py
```

也可以参考 `configs/example_source_papers.csv` 中的样例模板手动填写（填写完成后将文件名改为 `configs/source_papers.csv`）。

### Step 3. 基于标题与摘要的初筛

运行以下命令进行相似度计算和排序：

```bash
python src/rank.py
```
