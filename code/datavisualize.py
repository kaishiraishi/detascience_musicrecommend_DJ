import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
df = pd.read_csv('track_features.csv')

# 基本統計量を計算
stats = df.describe()

# プロットを行うための準備
fig, ax = plt.subplots()

# 各プレイリスト番号でループ
for playlist_number in df['playlist_number'].unique():
    playlist_data = df[df['playlist_number'] == playlist_number]
    # 各特徴量についてプロット
    for feature in ['danceability', 'energy', 'tempo', 'loudness']:
        ax.scatter(playlist_data['play_number'], playlist_data[feature], label=f"{feature} (Playlist {playlist_number})")

ax.set_xlabel("Play Number")
ax.set_ylabel("Feature Value")
ax.legend(title="Features")

plt.show()
