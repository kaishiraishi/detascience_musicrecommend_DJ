import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込みと前処理
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # 必要な特徴量を選択
    features = [ 'danceability' , 'acousticness' , 'tempo_scaled', 'instrumentalness' ,'loudness', 'key','speechiness' ,]
    return df[['track_name'] + features]

# ユークリッド距離の計算
def calculate_similarity(df_a, df_b):
    closest, _ = pairwise_distances_argmin_min(df_a.iloc[:, 1:], df_b.iloc[:, 1:])
    return closest

# K-means++ クラスタリング
def perform_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# 結果の可視化と分析
def analyze_clusters(data, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='danceability', y='acousticness', hue=labels, data=data, palette='viridis')
    plt.title('Cluster Visualization')
    plt.show()
    
    # クラスターのサンプル数
    cluster_counts = pd.Series(labels).value_counts()
    print("Cluster Sample Distribution:")
    print(cluster_counts)

    # クラスター内の楽曲名
    data['cluster'] = labels
    for c in sorted(data['cluster'].unique()):
        print(f"\nCluster {c} Tracks:")
        print(data[data['cluster'] == c]['track_name'].tolist())

# メイン関数
def main():
    df_a = load_and_preprocess('target_scaled.csv')
    df_b = load_and_preprocess('all_scaled.csv')
    
    # 類似度計算
    closest_indices = calculate_similarity(df_a, df_b)
    similar_tracks = df_b.iloc[closest_indices]
    
    # データの統合とクラスタリング
    combined_data = pd.concat([df_a, similar_tracks], ignore_index=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data.iloc[:, 1:])
    
    # 最適なクラスタ数の決定（エルボーメソッドは省略）
    labels, centers = perform_kmeans(scaled_data, num_clusters=3)
    
    # 結果の分析
    analyze_clusters(combined_data, labels)

if __name__ == '__main__':
    main()
