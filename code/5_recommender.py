import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# データの読み込み
a_df = pd.read_csv('./data/target_scaled.csv')  # 現在のプレイリストデータ
b_df = pd.read_csv('./data/all_scaled.csv')  # 比較対象の曲集合

# 除外するカラムを指定
excluded_columns = ['track_id', 'track_name', 'playlist_number', 'play_number']  # 除外したいカラム名のリスト

# データ正規化（除外カラムを除いて）
scaler = MinMaxScaler()
numerical_features = [col for col in a_df.columns if col not in excluded_columns]
a_df[numerical_features] = scaler.fit_transform(a_df[numerical_features])
b_df[numerical_features] = scaler.transform(b_df[numerical_features])

# A.csvの各曲に対してB.csvの曲とのコサイン類似度を計算
similarity_matrix = cosine_similarity(a_df[numerical_features], b_df[numerical_features])

# A.csvとB.csvで同じ曲を識別し、類似度行列から該当する類似度を0に設定
for i, a_track in enumerate(a_df['track_id']):
    for j, b_track in enumerate(b_df['track_id']):
        if a_track == b_track:
            similarity_matrix[i, j] = 0

# 類似度が最も高い上位数曲を選択
top_n = 10  # 上位5曲を選択する例
most_similar_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
recommended_songs = b_df.iloc[most_similar_indices.flatten()]

# IDが重複している曲を消去
recommended_songs = recommended_songs.drop_duplicates(subset='track_id', keep='first')

# 結果をC.csvに出力
recommended_songs.to_csv('Cluster.csv', index=False)

print(f"Recommendations based on cosine similarity for the top {top_n} similar songs have been saved to C.csv. Duplicate tracks have been removed.")
