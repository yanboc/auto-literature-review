import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import argparse
import json


class PaperSimilarity:
    def __init__(self, data_path, text_field="combined"):
        """
        初始化
        :param data_path: CSV 或 JSON 路径
        :param text_field: 使用 'title', 'abstract', 或 'combined'（默认）
        """
        self.data_path = data_path
        self.text_field = text_field
        self.df = self._load_data()
        self.texts = self._prepare_texts()

    def _load_data(self):
        if self.data_path.endswith(".csv"):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(".json"):
            df = pd.read_json(self.data_path, lines=False)
        else:
            raise ValueError("仅支持 .csv 或 .json 文件")
        assert "id" in df.columns and "title" in df.columns and "abstract" in df.columns
        return df

    def _prepare_texts(self):
        if self.text_field == "title":
            return self.df["title"].fillna("").tolist()
        elif self.text_field == "abstract":
            return self.df["abstract"].fillna("").tolist()
        elif self.text_field == "combined":
            # 标题权重更高：重复一次标题
            return (
                self.df["title"].fillna("")
                + " "
                + self.df["title"].fillna("")
                + " "
                + self.df["abstract"].fillna("")
            ).tolist()
        else:
            raise ValueError("text_field 必须是 'title', 'abstract', 或 'combined'")

    # ================ 方法1：TF-IDF 快速批量筛选 ================
    def compute_tfidf_similarity(self, top_k=10, threshold=0.5, output_path=None):
        """
        使用 TF-IDF + 余弦相似度快速计算相似论文
        :param top_k: 每篇论文返回最相似的 top_k 篇
        :param threshold: 相似度阈值（低于则忽略）
        :param output_path: 保存结果路径（JSON）
        :return: list of dict, each: {'id1', 'id2', 'similarity'}
        """
        print("🚀 正在计算 TF-IDF 相似度（适用于 10k+ 论文）...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),  # 包含 bigram 提升效果
        )
        tfidf_matrix = vectorizer.fit_transform(self.texts)
        print(f"TF-IDF 矩阵形状: {tfidf_matrix.shape}")

        # 计算稀疏矩阵的成对相似度（内存友好）
        # 注意：10k x 10k 矩阵需约 800MB 内存（float64），可接受
        cosine_sim = cosine_similarity(tfidf_matrix)
        results = []

        n = len(self.df)
        for i in tqdm(range(n), desc="筛选相似论文"):
            # 获取与论文 i 最相似的 top_k 篇（排除自己）
            sim_scores = cosine_sim[i]
            sim_scores[i] = -1  # 排除自身
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            for j in top_indices:
                sim = sim_scores[j]
                if sim >= threshold:
                    results.append(
                        {
                            "id1": self.df.iloc[i]["id"],
                            "id2": self.df.iloc[j]["id"],
                            "title1": self.df.iloc[i]["title"],
                            "title2": self.df.iloc[j]["title"],
                            "similarity": float(sim),
                            "method": "TF-IDF",
                        }
                    )
                else:
                    break  # 后续更小，可提前终止（因已排序）

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ TF-IDF 结果已保存至: {output_path}")

        return results

    # ================ 方法2：Sentence-BERT 高精度计算 ================
    def compute_sbert_similarity(
        self, top_k=5, threshold=0.6, model_name="all-MiniLM-L6-v2", output_path=None
    ):
        """
        使用 Sentence-BERT 计算高精度语义相似度
        :param top_k: 每篇返回 top_k 相似
        :param threshold: 相似度阈值
        :param model_name: SBERT 模型名称
        :param output_path: 保存路径
        :return: list of dict
        """
        print(f"🧠 正在加载 SBERT 模型: {model_name}...")
        model = SentenceTransformer(model_name)

        print("🔤 正在编码所有论文（标题+摘要）...")
        embeddings = model.encode(self.texts, show_progress_bar=True, batch_size=128)

        print("🔍 计算余弦相似度（高精度）...")
        cosine_sim = cosine_similarity(embeddings)
        results = []

        n = len(self.df)
        for i in tqdm(range(n), desc="推荐相似论文"):
            sim_scores = cosine_sim[i]
            sim_scores[i] = -1
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            for j in top_indices:
                sim = sim_scores[j]
                if sim >= threshold:
                    results.append(
                        {
                            "id1": self.df.iloc[i]["id"],
                            "id2": self.df.iloc[j]["id"],
                            "title1": self.df.iloc[i]["title"],
                            "title2": self.df.iloc[j]["title"],
                            "similarity": float(sim),
                            "method": "SBERT",
                        }
                    )
                else:
                    break

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ SBERT 结果已保存至: {output_path}")

        return results


# ================ 使用示例 ================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="论文相似度计算工具")
    parser.add_argument("--data", type=str, required=True, help="论文数据文件路径 (CSV/JSON)")
    parser.add_argument("--mode", type=str, choices=["tfidf", "sbert", "both"], default="both")
    parser.add_argument("--tfidf_out", type=str, default="tfidf_results.json")
    parser.add_argument("--sbert_out", type=str, default="sbert_results.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--tfidf_threshold", type=float, default=0.4)
    parser.add_argument("--sbert_threshold", type=float, default=0.6)

    args = parser.parse_args()

    # 初始化
    ps = PaperSimilarity(args.data, text_field="combined")

    if args.mode in ["tfidf", "both"]:
        tfidf_results = ps.compute_tfidf_similarity(
            top_k=args.top_k, threshold=args.tfidf_threshold, output_path=args.tfidf_out
        )
        print(f"🔍 TF-IDF 找到 {len(tfidf_results)} 对相似论文")

    if args.mode in ["sbert", "both"]:
        sbert_results = ps.compute_sbert_similarity(
            top_k=args.top_k, threshold=args.sbert_threshold, output_path=args.sbert_out
        )
        print(f"🧠 SBERT 找到 {len(sbert_results)} 对相似论文")
