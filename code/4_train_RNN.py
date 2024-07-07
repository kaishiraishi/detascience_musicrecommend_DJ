import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# データの読み込み
data = pd.read_csv('./data/track_features_0701_updated.csv')

# 必要のないカラムを削除
data = data.drop(columns=['track_id', 'track_name', 'duration_ms', 'tempo'])
print(data.head())

# プレイリスト番号でグループ化し、各プレイリストを独立したシーケンスとして扱う
grouped = data.groupby('playlist_number')

sequences = []
labels = []



# 10曲ごとの特徴量を入力とし、次の1曲の特徴量をラベルとする
for name, group in grouped:
    if len(group) < 11:
        continue
    for i in range(len(group) - 10):
        sequences.append(group.iloc[i:i+10, :-1].values)
        labels.append(group.iloc[i+10, :-1].values)

# データをnumpy配列に変換
X = np.array(sequences)
y = np.array(labels)

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


#https://open.spotify.com/playlist/54PDYvBhOWsLQdNx0QBfir?si=cfc051549684486a
# モデルの構築
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(y_train.shape[1])
])

# モデルのコンパイル
model.compile(optimizer=Adam(), loss='mean_squared_error')

# モデルの訓練
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# モデルの保存
model.save('./model/1rnn_model.h5')
