import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
from tqdm import tqdm
import pickle

def load_json_data(file_path):
    """加载JSON数据并提取context和image_id"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤掉没有context字段的样本
    filtered_data = []
    for item in data:
        if "context" in item and "image_id" in item:
            filtered_data.append({
                "context": item["context"],
                "image_id": item["image_id"]
            })
    
    return filtered_data

def encode_texts(texts, model_name, save_dir=None):
    """使用sentence-bert模型对文本进行编码"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Encoding texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 保存embeddings如果提供了保存路径
    if save_dir:
        save_path = os.path.join(save_dir, "embeddings.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {save_path}")
    
    return embeddings
def determine_optimal_k(embeddings, max_k=15, output_dir=None):
    """使用肘部法则和轮廓分数确定最佳的K值"""
    print("Determining optimal K...")
    inertia_values = []
    silhouette_values = []
    k_values = list(range(2, max_k + 1))
    
    for k in tqdm(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
        
        if k > 1:  # 轮廓分数至少需要2个聚类
            silhouette_values.append(silhouette_score(embeddings, kmeans.labels_))
    
    # 绘制肘部图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia_values, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # 确保x和y维度一致 - 这里是关键修复
    silhouette_k_values = k_values  # 复制一份k_values
    if len(silhouette_values) < len(silhouette_k_values):
        # 如果silhouette_values比k_values短，裁剪k_values
        silhouette_k_values = silhouette_k_values[:len(silhouette_values)]
    
    plt.subplot(1, 2, 2)
    plt.plot(silhouette_k_values, silhouette_values, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    
    plt.tight_layout()
    
    # 保存图像到指定目录或当前目录
    save_path = os.path.join(output_dir if output_dir else '.', 'optimal_k.png')
    plt.savefig(save_path)
    plt.close()
    
    # 找到最佳K值（修复这里避免索引错误）
    best_silhouette_index = silhouette_values.index(max(silhouette_values))
    optimal_k = silhouette_k_values[best_silhouette_index]
    print(f"Optimal K value: {optimal_k}")
    return optimal_k

def perform_clustering(embeddings, n_clusters):
    """执行K-means聚类"""
    print(f"Performing K-means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

def visualize_clusters(embeddings, clusters, image_ids, output_dir=None):
    """使用PCA可视化聚类结果"""
    print("Visualizing clusters using PCA...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 创建DataFrame以便绘图
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': clusters,
        'image_id': image_ids
    })
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', alpha=0.7)
    plt.title('Clusters visualization using PCA')
    
    # 保存图像到指定目录或当前目录
    save_path = os.path.join(output_dir if output_dir else '.', 'clusters_visualization.png')
    plt.savefig(save_path)
    plt.close()
    
    return df

def save_results(df, output_path, samples_per_cluster=None):
    """
    保存聚类结果，按照聚类大小降序排列，只包含image_id、text和cluster三列
    如果samples_per_cluster不为None，还会额外保存每个类别的n个样本
    """
    # 统计每个聚类的样本数量并按数量降序排序
    cluster_stats = df['cluster'].value_counts().sort_values(ascending=False)
    
    # 创建一个新的空DataFrame来存储排序后的结果
    sorted_df = pd.DataFrame(columns=['image_id', 'text', 'cluster'])
    
    # 如果需要提取样本，创建一个DataFrame来存储
    if samples_per_cluster is not None:
        sampled_df = pd.DataFrame(columns=['image_id', 'text', 'cluster'])
    
    # 按照聚类大小降序遍历每个聚类
    for cluster_id in cluster_stats.index:
        # 获取当前聚类的所有样本
        cluster_data = df[df['cluster'] == cluster_id][['image_id', 'text', 'cluster']]
        # 添加到新的DataFrame
        sorted_df = pd.concat([sorted_df, cluster_data])
        
        # 如果需要提取样本，从每个聚类中提取n个
        if samples_per_cluster is not None:
            n_samples = min(samples_per_cluster, len(cluster_data))
            sampled_df = pd.concat([sampled_df, cluster_data.head(n_samples)])
    
    # 保存为CSV
    sorted_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # 保存抽样数据
    if samples_per_cluster is not None:
        sample_output_path = output_path.replace('.csv', f'_sampled_{samples_per_cluster}perCluster.csv')
        sampled_df.to_csv(sample_output_path, index=False)
        print(f"Sampled results ({samples_per_cluster} per cluster) saved to {sample_output_path}")
    
    # 打印聚类统计信息
    print("\nCluster statistics:")
    for cluster, count in cluster_stats.items():
        print(f"Cluster {cluster}: {count} samples")

def main():
    # 参数设置
    input_file = "/Users/xiaomingcheng/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-dev-mrc-updated.json"
    output_dir = "CoT-sample-selection/cluster"
    model_name = "CoT-sample-selection/cluster/model"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)
    texts = [item["context"] for item in data]
    image_ids = [item["image_id"] for item in data]
    
    print(f"Loaded {len(texts)} text samples")
    
    # 从缓存文件加载embeddings
    try:
        with open(os.path.join('CoT-sample-selection/cluster', "embeddings.pkl"), 'rb') as f:
            embeddings = pickle.load(f)
            print(f"Loaded embeddings with shape {embeddings.shape}")
    except FileNotFoundError:
        print("Embeddings file not found. Generating new embeddings...")
        embeddings = encode_texts(texts, model_name, output_dir)
    
    # 确定最佳K值
    # optimal_k = determine_optimal_k(embeddings, output_dir=output_dir)
    
    # 执行聚类
    print("Performing clustering...")
    optimal_k = 30  # 使用默认值10
    clusters = perform_clustering(embeddings, optimal_k)
    
    # 可视化并保存结果
    print("Visualizing clusters...")
    df = visualize_clusters(embeddings, clusters, image_ids, output_dir)
    df['text'] = texts  # 添加原始文本
    
    # 保存结果
    output_path = os.path.join(output_dir, f"twitter_clusters_k{optimal_k}.csv")
    save_results(df, output_path, samples_per_cluster=1)
    
    print("Clustering complete!")

if __name__ == "__main__":
    main()