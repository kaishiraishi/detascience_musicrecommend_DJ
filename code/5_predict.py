import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# モデルの読み込み
model = load_model('./model/rnn_model.h5')

# 新しいデータの読み込み
df_new_tracks = pd.read_csv('./data/ALL_track_features_updated.csv')

df_new_tracks = df_new_tracks.drop(columns=['track_id', 'track_name', 'play_number'])

print(df_new_tracks.head())


# 欠損値の確認
#print(df_new_tracks.isnull().sum())

# 欠損値があれば、例えば列の平均値で埋める
#df_new_tracks.fillna(df_new_tracks.mean(), inplace=True)

# 不要な列があれば除外（例えばIDや名前など）
df_new_tracks = df_new_tracks.drop(columns=['track_id', 'track_name', 'playlist_number'])

# データの欠損値を平均値で補完
df_new_tracks.fillna(df_new_tracks.mean(), inplace=True)

# 特徴量の選択とNumPy配列に変換
features = df_new_tracks.values

# MinMaxScalerインスタンスの作成（訓練データでfitしたスケーラーを使用することが理想）
scaler = MinMaxScaler()
scaler.fit(features)  # 実際には訓練データでfitしたスケーラーを使用するべき

# データをシーケンスに変換する関数
def create_sequences(features, sequence_length):
    xs = []
    for i in range(len(features) - sequence_length + 1):
        xs.append(features[i:(i + sequence_length)])
    return np.array(xs)

# シーケンスの長さを定義
sequence_length = 10
X_new = create_sequences(features, sequence_length)

# 予測を実行
predictions = model.predict(X_new)

# 予測結果をDataFrameに格納
results_df = pd.DataFrame(predictions, columns=['Predicted_Play_Number'])

# 予測結果のDataFrameをCSVファイルとして保存
results_df.to_csv('./data/predicted_play_numbers.csv', index=False)

print("予測完了し、結果を 'predicted_play_numbers.csv' に保存しました！")



