import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# データの読み込み
a_df = pd.read_csv('./data/all_scaled.csv')  # 現在のプレイリストデータ

# データ正規化
scaler = MinMaxScaler()
numerical_features = a_df.columns[2:]  # 最初の2カラムを除いたすべての数値特徴
a_df[numerical_features] = scaler.fit_transform(a_df[numerical_features])

# Elbow法による最適なクラスタ数の決定
SSE = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(a_df[numerical_features])
    SSE.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), SSE, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()
