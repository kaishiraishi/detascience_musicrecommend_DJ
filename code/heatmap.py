import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み
df = pd.read_csv('./data/target_scaled.csv')

# トラックIDとトラック名を除外
df_numeric = df.drop(columns=['track_id', 'track_name'])

print(df_numeric.head())

# プレイリスト番号によってデータを分割
playlist_groups = df_numeric.groupby('playlist_number')

# 各プレイリストの相関行列のヒートマップを表示
for name, group in playlist_groups:
    # 数値データのみを対象に相関行列を計算
    correlation = group.drop(columns=['playlist_number']).corr()

    # 相関行列のヒートマップを表示
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Feature Correlation in Playlist {name}')
    plt.show()
