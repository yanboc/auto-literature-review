from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).parent.parent


class Paper(pd.DataFrame):
    """
    
    
    Columns
        - title: title of the paper (str)
        - authors: authors of the paper (List[str])
        - abstract: abstract of the paper (str)
        - pub_name: name of the publication venue (str)
        - pub_year: year of the publication (str)
        - extra_id: extra identifier dictionary (Dict[str])，如 {'ArXiv': 'xxx', 'DOI': 'xxx'}
    """
    _required_columns = {
        'title', 'authors', 'abstract', 'pub_name', 'pub_year', 'extra_id'
    }

    def __init__(self, data: Optional[Dict] = None, **kwargs):
        if data is None:
            data = {col: [] for col in self._required_columns}
        
        # 转换数据为字典（支持列表嵌套字典等格式）
        if not isinstance(data, dict):
            # 尝试从非字典数据（如列表）构建临时字典
            temp_df = pd.DataFrame(data,** kwargs)
            data = {col: temp_df[col].tolist() for col in temp_df.columns}
        
        # 校验列名是否符合要求
        input_columns = set(data.keys())
        missing_cols = self._required_columns - input_columns
        extra_cols = input_columns - self._required_columns
        
        if missing_cols:
            raise ValueError(f"缺少必填列: {missing_cols}")
        if extra_cols:
            raise ValueError(f"不允许的额外列: {extra_cols}")
        
        # 调用父类构造函数
        super().__init__(data, **kwargs)

    # 可选：添加类型提示（增强IDE自动补全）
    @property
    def _constructor(self):
        return PaperDataFrame