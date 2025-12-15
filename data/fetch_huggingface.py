"""
从 Hugging Face datasets 和 spaces 获取数据并转换为 JSON 格式
"""
import json
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("警告: datasets 库未安装，请运行: pip install datasets")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("警告: requests 库未安装，请运行: pip install requests")


def extract_dataset_name(url: str) -> Optional[str]:
    """
    从 Hugging Face URL 中提取 dataset 或 space 的名称
    
    Args:
        url: Hugging Face URL (dataset 或 space)
        
    Returns:
        dataset/space 名称，格式为 "username/dataset_name" 或 None
    """
    # 匹配 datasets 或 spaces 的 URL 模式
    pattern = r'huggingface\.co/(?:datasets|spaces)/([^/]+/[^/?#]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def fetch_dataset_data(dataset_name: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从 Hugging Face dataset 获取数据
    
    Args:
        dataset_name: dataset 名称，格式为 "username/dataset_name"
        split: 数据集分割名称（如 'train', 'test'），如果为 None 则获取所有分割
        
    Returns:
        数据列表，每个元素是一个字典
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets 库未安装，请运行: pip install datasets")
    
    try:
        # 加载数据集
        dataset = load_dataset(dataset_name)
        
        # 如果指定了 split，只获取该分割
        if split:
            if split in dataset:
                data = dataset[split]
            else:
                raise ValueError(f"Split '{split}' 不存在。可用的 splits: {list(dataset.keys())}")
        else:
            # 如果没有指定 split，获取第一个可用的 split
            if len(dataset) == 0:
                raise ValueError("数据集为空")
            first_split = list(dataset.keys())[0]
            data = dataset[first_split]
        
        # 转换为列表格式
        return data.to_list()
    
    except Exception as e:
        raise Exception(f"获取 dataset '{dataset_name}' 失败: {str(e)}")


def fetch_space_data(space_name: str) -> Dict[str, Any]:
    """
    从 Hugging Face space 获取数据（通过 API）
    
    Args:
        space_name: space 名称，格式为 "username/space_name"
        
    Returns:
        包含 space 信息的字典
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests 库未安装，请运行: pip install requests")
    
    try:
        # 使用 Hugging Face API 获取 space 信息
        api_url = f"https://huggingface.co/api/spaces/{space_name}"
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        space_info = response.json()
        
        # 如果 space 有 dataset，尝试获取 dataset 数据
        if 'datasets' in space_info and space_info['datasets']:
            # 尝试获取关联的 dataset
            dataset_name = space_info['datasets'][0]
            try:
                dataset_data = fetch_dataset_data(dataset_name)
                return {
                    "space_info": space_info,
                    "associated_dataset": dataset_name,
                    "data": dataset_data
                }
            except Exception as e:
                return {
                    "space_info": space_info,
                    "error": f"无法获取关联数据集: {str(e)}"
                }
        
        return {
            "space_info": space_info,
            "note": "此 space 没有关联的数据集，返回的是 space 的元数据信息"
        }
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"获取 space '{space_name}' 失败: {str(e)}")


def fetch_huggingface_data(url: str, split: Optional[str] = None) -> Dict[str, Any]:
    """
    从 Hugging Face URL (dataset 或 space) 获取数据并转换为 JSON 格式
    
    Args:
        url: Hugging Face URL (dataset 或 space)
        split: 对于 datasets，指定要获取的分割（可选）
        
    Returns:
        包含数据的字典，可以转换为 JSON
    """
    dataset_name = extract_dataset_name(url)
    
    if not dataset_name:
        raise ValueError(f"无法从 URL 中提取 dataset/space 名称: {url}")
    
    # 判断是 dataset 还是 space
    if "/datasets/" in url:
        # 这是 dataset
        data = fetch_dataset_data(dataset_name, split=split)
        return {
            "type": "dataset",
            "name": dataset_name,
            "url": url,
            "data": data,
            "count": len(data)
        }
    elif "/spaces/" in url:
        # 这是 space
        data = fetch_space_data(dataset_name)
        return {
            "type": "space",
            "name": dataset_name,
            "url": url,
            **data
        }
    else:
        raise ValueError(f"无法识别 URL 类型: {url}")


def save_to_json(data: Dict[str, Any], output_file: str, indent: int = 2) -> None:
    """
    将数据保存为 JSON 文件
    
    Args:
        data: 要保存的数据字典
        output_file: 输出文件路径
        indent: JSON 缩进空格数
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"数据已保存到: {output_file}")


# 示例使用
if __name__ == "__main__":
    # 示例 1: 从 dataset 获取数据
    dataset_url = "https://huggingface.co/datasets/DeepNLP/NIPS-2022-Accepted-Papers"
    print(f"正在获取 dataset 数据: {dataset_url}")
    try:
        dataset_result = fetch_huggingface_data(dataset_url)
        print(f"成功获取 {dataset_result['count']} 条记录")
        print(f"前 3 条数据示例:")
        for i, item in enumerate(dataset_result['data'][:3], 1):
            print(f"\n记录 {i}:")
            print(json.dumps(item, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例 2: 从 space 获取数据
    space_url = "https://huggingface.co/spaces/ICML2022/ICML2022_papers"
    print(f"正在获取 space 数据: {space_url}")
    try:
        space_result = fetch_huggingface_data(space_url)
        print(f"Space 信息:")
        print(json.dumps(space_result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"错误: {e}")

