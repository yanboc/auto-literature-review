"""
批量从 urls.py 中的所有 Hugging Face URLs 获取数据
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional
from fetch_huggingface import fetch_huggingface_data, save_to_json
from urls import urls


def batch_fetch_all(output_dir: str = "output", split: Optional[str] = None):
    """
    批量获取所有 URLs 中的数据
    
    Args:
        output_dir: 输出目录
        split: 对于 datasets，指定要获取的分割（可选）
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    
    for key, url in urls.items():
        print(f"\n{'='*60}")
        print(f"正在处理: {key}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        
        try:
            data = fetch_huggingface_data(url, split=split)
            
            # 保存单个文件
            output_file = output_path / f"{key}.json"
            save_to_json(data, str(output_file))
            
            results[key] = {
                "status": "success",
                "url": url,
                "output_file": str(output_file),
                "count": data.get("count", "N/A")
            }
            
            print(f"✓ 成功: {key}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ 失败: {key} - {error_msg}")
            results[key] = {
                "status": "error",
                "url": url,
                "error": error_msg
            }
    
    # 保存汇总结果
    summary_file = output_path / "summary.json"
    save_to_json(results, str(summary_file))
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("汇总结果:")
    print(f"{'='*60}")
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"汇总文件: {summary_file}")


if __name__ == "__main__":
    # 可以指定 split 参数
    split = None
    if len(sys.argv) > 1:
        split = sys.argv[1]
        print(f"使用指定的 split: {split}")
    
    batch_fetch_all(split=split)

