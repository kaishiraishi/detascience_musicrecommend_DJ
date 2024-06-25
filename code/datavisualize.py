import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('track_features.csv')

# play_numberと他の数値特徴量との関係を散布図で表示
fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # サブプロットの準備
fig.subplots_adjust(hspace=0.5, wspace=0.3)
axes = axes.ravel()

numerical_features = df.columns[2:-1]  # 'play_number'を除く数値特徴量を選択
for idx, col in enumerate(numerical_features):
    axes[idx].scatter(df['play_number'], df[col], alpha=0.5, color='blue')
    axes[idx].set_title(f'{col} vs play_number')
    axes[idx].set_xlabel('Play Number')
    axes[idx].set_ylabel(col)

# 不要なプロット領域を非表示
for idx in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()
