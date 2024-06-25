import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# CSVファイルの読み込み
data = pd.read_csv('playlist_audio_features.csv')

# データの先頭を表示して確認
print(data.head())

# 目的変数と特徴量の選択
X = data[['instrumentalness', 'energy', 'acousticness', 'danceability']]
y = data['key']

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレスト回帰モデルのインスタンスを作成
forest = RandomForestRegressor(n_estimators=100, random_state=42)

# モデルのトレーニング
forest.fit(X_train, y_train)

# テストデータでの予済
y_pred = forest.predict(X_test)

# モデルのパフォーマンスを評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# 特徴量の重要度を取得
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# 特徴量の重要度を表示
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
