import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# データの読み込み
df = pd.read_csv('./data/track_features_0701_updated.csv')




df = df.drop(columns=['track_id', 'track_name'])

# 欠損値の確認
print(df.isnull().sum())

# 欠損値があれば、例えば列の平均値で埋める
df.fillna(df.mean(), inplace=True)


# 既に正規化された特徴量を使用
features = df.drop(['play_number'], axis=1)  # play_numberを除外
target = df['play_number'].values.reshape(-1, 1)  # ターゲットを適切な形状に変換

# データをシーケンスに変換する関数
def create_sequences(features, target, sequence_length):
    xs = []
    ys = []
    for i in range(len(features) - sequence_length):
        xs.append(features[i:(i + sequence_length)])
        ys.append(target[i + sequence_length])
    return np.array(xs), np.array(ys)

# 特徴量とターゲットの準備
features = df.drop(['play_number'], axis=1).values
target = df['play_number'].values.reshape(-1, 1)

# シーケンス生成
sequence_length = 10
X, y = create_sequences(features, target, sequence_length)

# データを訓練用とテスト用に分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# モデルの設計
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

# モデルのコンパイル
model.compile(loss='mean_squared_error', optimizer='adam')

# モデルの学習
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# モデルの評価
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss}')


# モデルの保存
model.save('./model/rnn_model.h5')