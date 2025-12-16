# 自动文献综述助手

![Language: Python](https://img.shields.io/badge/Language-Python-blue?logo=python) ![Subject: CS/ML/AI](https://img.shields.io/badge/Subject-CS/ML/AI-yellowgreen) ![Model: Qwen-3](https://img.shields.io/badge/Model-Qwen--3-ff69b4)

大家好🤗我做了一个集成 AI 功能的自动文献综述助手，能够根据几个关键词或参考文献自动生成一份文献清单并自动整理/保存 AI 生成的文献总结。而且更重要的是：你可以自由地往里面加入你需要的功能，因为整个项目的代码非常简单（~200行）。当前版本的主要特性包括：

1. **（获取元数据）** 我们维护了一份从 Hugging Face 上获取的“历届AI会议接收论文清单”。基于这份清单，我们将接收论文的元信息（包括题目、作者、摘要、PDF、关键词）清洗并整理成 JSON 格式，方便后续自定义分析。
2. **（基于规则的初筛）** 我们基于关键词匹配、摘要的相似度和作者关系网分析等方式对文章进行初筛。另外，我们还
3. **（AI/Agentic 功能）** 对于初筛过后的文章，我们使用 AI 功能会 PDF 全文进行总结（输出为`markdown`格式），并对总结过后的文章进行进一步比对，从而得出一份

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法 1: 单个 URL 获取数据

```python
from fetch_huggingface import fetch_huggingface_data, save_to_json

# 从 dataset 获取数据
url = "https://huggingface.co/datasets/DeepNLP/NIPS-2022-Accepted-Papers"
data = fetch_huggingface_data(url)

# 保存为 JSON 文件
save_to_json(data, "output.json")

# 或者直接使用 JSON
import json
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)
```

### 方法 2: 批量获取所有 URLs 的数据

```bash
# 获取所有 URLs 的数据
python data/batch_fetch.py

# 指定 dataset 的 split（如 'train', 'test'）
python data/batch_fetch.py train
```

批量处理会在 `output/` 目录下生成：
- 每个 URL 对应的 JSON 文件（如 `neurips_2022.json`）
- `summary.json` 汇总文件，包含所有处理结果

### 方法 3: 在代码中使用

```python
from fetch_huggingface import fetch_huggingface_data
import json

# 获取 dataset 数据
dataset_url = "https://huggingface.co/datasets/DeepNLP/NIPS-2022-Accepted-Papers"
result = fetch_huggingface_data(dataset_url)

# 访问数据
print(f"数据类型: {result['type']}")
print(f"数据条数: {result['count']}")
print(f"前 3 条数据:")
for item in result['data'][:3]:
    print(json.dumps(item, ensure_ascii=False, indent=2))
```

## 函数说明

### `fetch_huggingface_data(url, split=None)`

从 Hugging Face URL 获取数据。

**参数:**
- `url`: Hugging Face dataset 或 space 的 URL
- `split`: 对于 datasets，可以指定要获取的分割（如 'train', 'test'），默认为 None（获取第一个可用分割）

**返回:**
- 包含数据的字典，可以转换为 JSON

**示例:**
```python
# 获取整个 dataset
data = fetch_huggingface_data("https://huggingface.co/datasets/...")

# 获取特定 split
data = fetch_huggingface_data("https://huggingface.co/datasets/...", split="train")
```

### `save_to_json(data, output_file, indent=2)`

将数据保存为 JSON 文件。

**参数:**
- `data`: 要保存的数据字典
- `output_file`: 输出文件路径
- `indent`: JSON 缩进空格数（默认 2）

## 注意事项

1. **Datasets**: 使用 `datasets` 库直接加载，支持所有 Hugging Face datasets
2. **Spaces**: 通过 Hugging Face API 获取 space 信息，如果 space 关联了 dataset，会尝试获取关联的数据
3. **网络连接**: 需要能够访问 Hugging Face 网站
4. **数据大小**: 大型数据集可能需要较长时间下载和处理

## 输出格式

### Dataset 输出格式

```json
{
  "type": "dataset",
  "name": "username/dataset_name",
  "url": "https://huggingface.co/datasets/...",
  "data": [
    {
      "field1": "value1",
      "field2": "value2"
    }
  ],
  "count": 100
}
```

### Space 输出格式

```json
{
  "type": "space",
  "name": "username/space_name",
  "url": "https://huggingface.co/spaces/...",
  "space_info": {
    "id": "...",
    "title": "...",
    ...
  },
  "associated_dataset": "username/dataset_name",
  "data": [...]
}
```

