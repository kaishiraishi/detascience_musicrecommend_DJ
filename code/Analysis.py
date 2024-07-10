import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import ccf

# 1. データの準備
# CSVファイルからデータを読み込む
data = pd.read_csv('./data/target_scaled.csv')
data['play_number'] = pd.to_datetime(data['play_number'])  # play_number列をdatetime型に変換
data.set_index('play_number', inplace=True)  # play_numberをインデックスに設定

# 2. 自己相関分析
# テンポの自己相関をプロット
plt.figure(figsize=(10, 5))
plot_acf(data['tempo'], lags=20)
plt.title('Auto-Correlation of Tempo')
plt.show()


# 3. クロス相関分析
# テンポとエネルギーのクロス相関を計算
cross_correlation = ccf(data['tempo'], data['energy'])
plt.figure(figsize=(10, 5))
plt.stem(cross_correlation[:20], use_line_collection=True)  # 最初の20ラグのクロス相関をプロット
plt.title('Cross-Correlation between Tempo and Energy')
plt.show()

# 4. 結果の解釈
# 結果の解釈は、プロットから読み取ることができます。特に、自己相関とクロス相関の両方で顕著なラグを識別し、これらが楽曲特徴の動的な関係を示しています。
