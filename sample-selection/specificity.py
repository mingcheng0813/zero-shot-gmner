
import os
import json
import numpy as np
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DomainSpecificityAnalyzer:
    """领域特异性分析器：评估句子与社交媒体语言风格的相似度"""
    
    def __init__(self):
        """初始化分析器，加载NLP模型和情感分析器"""
        self.nlp = spacy.load('en_core_web_sm')
        self.analyzer = SentimentIntensityAnalyzer()
    
    def extract_syntactic_features(self, sentence):
        """提取句子的句法特征"""
        doc = self.nlp(sentence)
        num_tokens = len(doc)  # 句子长度
        num_clauses = sum(1 for token in doc if token.dep_ == 'ROOT')  # 从句数量
        
        # 处理可能的空文档情况
        if len(doc) > 0:
            tree_depth = max([token.i for token in doc if token.dep_ == 'ROOT'], default=0)  # 树深度
        else:
            tree_depth = 0
            
        return num_tokens, num_clauses, tree_depth
    
    def get_sentiment_scores(self, sentence):
        """获取句子的情感分数"""
        return self.analyzer.polarity_scores(sentence)
    
    def calculate_ttr(self, sentence):
        """计算类符-形符比率(Type-Token Ratio)，衡量词汇丰富度"""
        words = sentence.lower().split()
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if len(words) > 0 else 0
        return ttr
    
    def calculate_syntactic_similarity(self, sample_sentence, reference_sentences):
        """计算一个样本句子与参考句子列表的句法相似度"""
        sample_features = self.extract_syntactic_features(sample_sentence)
        distances = []
        
        for ref_sentence in reference_sentences:
            ref_features = self.extract_syntactic_features(ref_sentence)
            distance = sum((sf - rf) ** 2 for sf, rf in zip(sample_features, ref_features)) ** 0.5
            distances.append(distance)
            
        return distances

    def calculate_lexical_similarity(self, sample_sentence, reference_sentences, tfidf=None, ref_tfidf=None):
        """计算一个样本句子与参考句子列表的词汇相似度"""
        # 如果没有提供TF-IDF模型，则创建一个新的
        if tfidf is None:
            tfidf = TfidfVectorizer().fit(reference_sentences)
            ref_tfidf = tfidf.transform(reference_sentences)
        
        # 计算词汇相似度
        sample_tfidf = tfidf.transform([sample_sentence])
        cosine_sims = cosine_similarity(sample_tfidf, ref_tfidf).flatten()
        
        # 计算TTR
        sample_ttr = self.calculate_ttr(sample_sentence)
        ref_ttrs = [self.calculate_ttr(sent) for sent in reference_sentences]
        ttr_diffs = [abs(sample_ttr - ref_ttr) for ref_ttr in ref_ttrs]
        
        # 组合词汇相似度和TTR差异
        combined_scores = [
            (0.7 * (1 - sim) + 0.3 * diff)  # 将相似度转换为差异值
            for sim, diff in zip(cosine_sims, ttr_diffs)
        ]
        
        return combined_scores
    
    def get_similarity_scores(self, sample_sentence, reference_sentences, 
                              syntactic_weight=0.5, lexical_weight=0.5,
                              tfidf=None, ref_tfidf=None):
        """计算一个样本句子与所有参考句子的综合相似度"""
        # 计算句法和词汇相似度
        syntactic_diffs = self.calculate_syntactic_similarity(sample_sentence, reference_sentences)
        lexical_diffs = self.calculate_lexical_similarity(sample_sentence, reference_sentences, tfidf, ref_tfidf)
        
        # 归一化句法差异分数（转换为0-1范围）
        if len(syntactic_diffs) > 0:
            max_syntactic = max(syntactic_diffs)
            min_syntactic = min(syntactic_diffs)
            if max_syntactic > min_syntactic:
                syntactic_scores = [(d - min_syntactic) / (max_syntactic - min_syntactic) for d in syntactic_diffs]
            else:
                syntactic_scores = [0.5 for _ in syntactic_diffs]
        else:
            syntactic_scores = []
        
        # 计算综合得分（差异越小，得分越高）
        combined_scores = []
        for i in range(len(reference_sentences)):
            weighted_diff = syntactic_weight * syntactic_scores[i] + lexical_weight * lexical_diffs[i]
            similarity = 1 - weighted_diff  # 转换为相似度
            combined_scores.append(similarity)
            
        return combined_scores


def load_test_data(test_data_path):
    """加载测试数据集"""
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤掉没有context字段的条目
    filtered_data = []
    for item in data:
        if "context" in item:
            filtered_data.append(item)
    
    return filtered_data


def load_cluster_samples(cluster_samples_path):
    """加载聚类样本CSV文件"""
    df = pd.read_csv(cluster_samples_path)
    cluster_samples = []
    
    for _, row in df.iterrows():
        cluster_samples.append({
            'image_id': row['image_id'],
            'text': row['text'],
            'cluster': row['cluster']
        })
    
    return cluster_samples


def find_best_matches(test_data, cluster_samples, analyzer):
    """为每条测试数据找出最匹配的聚类样本"""
    # 准备参考句子列表
    reference_sentences = [sample['text'] for sample in cluster_samples]
    
    # 预计算TF-IDF模型和矩阵
    tfidf = TfidfVectorizer().fit(reference_sentences)
    ref_tfidf = tfidf.transform(reference_sentences)
    
    matches = []
    total = len(test_data)
    
    # 为每条测试数据找最匹配的样本
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"处理进度: {i}/{total}")
            
        if "context" not in item:
            # 如果没有context字段，添加空匹配结果
            matches.append({**item})
            continue
            
        # 计算测试样本与所有聚类样本的相似度
        similarity_scores = analyzer.get_similarity_scores(
            item["context"], reference_sentences, 
            tfidf=tfidf, ref_tfidf=ref_tfidf
        )
        
        # Sort all samples by similarity score in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
        sorted_scores = [similarity_scores[i] for i in sorted_indices]
        sorted_image_ids = [cluster_samples[i]["image_id"] for i in sorted_indices]
        
        # Add all matches in order of similarity to the result
        matches.append({
            **item,
            "matched_image_ids": sorted_image_ids,
            "similarity_scores": [float(score) for score in sorted_scores]
        })
    
    return matches


def main():
    """主函数"""
    # 配置文件路径
    test_data_path = "/Users/xxxxxx/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated.json"
    cluster_samples_path = "CoT-sample-selection/cluster/twitter_clusters_k20_sampled_1perCluster.csv"
    output_path = "/Users/xxxxxx/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated-with-matches.json"
    
    # 加载数据
    print("加载测试数据...")
    test_data = load_test_data(test_data_path)
    print(f"已加载 {len(test_data)} 条测试数据")
    
    print("加载聚类样本...")
    cluster_samples = load_cluster_samples(cluster_samples_path)
    print(f"已加载 {len(cluster_samples)} 个聚类样本")
    
    # 初始化分析器
    print("初始化分析器...")
    analyzer = DomainSpecificityAnalyzer()
    
    # 寻找最佳匹配
    print("开始计算相似度并寻找最佳匹配...")
    matched_data = find_best_matches(test_data, cluster_samples, analyzer)
    
    # 保存结果
    print(f"保存结果到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(matched_data, f, ensure_ascii=False, indent=2)
    
    print("处理完成!")


if __name__ == "__main__":
    main()