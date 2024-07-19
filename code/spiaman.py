# 必要なライブラリのインポート
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('./data/target_scaled.csv')

# 特徴量リストの定義（'track_id'と'track_name'は相関分析から除外）
features = ['danceability', 'acousticness', 'tempo_scaled', 'instrumentalness',
            'energy', 'key', 'speechiness', 'mode', 'valence', 'liveness', 'play_number']

# 空のデータフレームを用意
correlation_data = pd.DataFrame(index=features, columns=features)

# スピアマン相関係数の計算
for feature1 in features:
    for feature2 in features:
        if feature1 == feature2:
            # 同一の特徴量の場合は相関係数を1とする
            correlation_data.loc[feature1, feature2] = 1.0
        else:
            # スピアマンの順位相関係数を計算
            corr, _ = spearmanr(df[feature1], df[feature2])
            correlation_data.loc[feature1, feature2] = corr

# 相関データフレームの値を数値型に変換
correlation_matrix = correlation_data.astype(float)

# ヒートマップの作成
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Rank Correlation Heatmap')
plt.show()
