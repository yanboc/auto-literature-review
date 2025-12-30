import pytest
from src.parse_sources import parse_from_semantic_scholar, parse_from_arxiv

# test parse_from_semantic_scholar function
def test_parse_from_semantic_scholar_with_title():
    key_type = "title"
    key_value = "Feature Purification: How Adversarial Training Performs Robust Deep Learning"
    paper = parse_from_semantic_scholar("Feature Purification: How Adversarial Training Performs Robust Deep Learning")
    
    assert paper["title"] == key_value

# test parse_from_arxiv function
def test_parse_from_arxiv():
    test_arxiv_ids = ["1512.03385", "1706.03762", "2005.14165"]
    for arxiv_id in test_arxiv_ids:
        paper = parse_from_arxiv(arxiv_id)
        assert paper["extra_id"]["arxiv_id"] == arxiv_id
        assert paper["title"] is not None
        assert paper["abstract"] is not None