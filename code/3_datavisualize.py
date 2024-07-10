import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み
#/Users/shiraishikai/Documents/GitHub/detascience_musicrecommend_DJ/data/ALL_track_features_updated.csv
#track_features_0701_updated.csv


df = pd.read_csv('./data/target_scaled.csv')

# トラックIDとトラック名を除外
df_numeric = df.drop(columns=['track_id', 'track_name'])
print("数値データの先頭:")
print(df_numeric.head())

# 基本統計量の表示
print("基本統計量:")
print(df_numeric.describe())

# 相関行列の計算
correlation = df_numeric.corr()

# 相関行列のヒートマップを表示
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation')
plt.show()

# プレイリスト番号によってデータを分割
playlist_groups = df.groupby('playlist_number')

# 各プレイリストでのデータ変化をプロット
for name, group in playlist_groups:
    plt.figure(figsize=(10, 6))
    plt.plot(group['play_number'], group['danceability'], label='Danceability')
    plt.plot(group['play_number'], group['energy'], label='Energy')
    plt.plot(group['play_number'], group['tempo_scaled'], label='Tempo_scaled')
    plt.plot(group['play_number'], group['acousticness'], label='Acousticness')
    plt.plot(group['play_number'], group['instrumentalness'], label='Instrumentalness')
    #plt.plot(group['play_number'], group['loudness'], label='Loudness')
    plt.plot(group['play_number'], group['liveness'], label='Liveness')
    plt.plot(group['play_number'], group['valence'], label='Valence')
    plt.plot(group['play_number'], group['speechiness'], label='Speechiness')
    
    plt.title(f'Feature Changes in Playlist {name}')
    plt.xlabel('Play Number')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.show()
