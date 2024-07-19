# 必要なライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# CSVファイルの読み込み
df = pd.read_csv('./data/target_scaled.csv')

# データの整理
df.sort_values('play_number', inplace=True)

# 第1グループの特徴量
features_group1 = ['danceability', 'acousticness', 'tempo', 'instrumentalness', 'energy']

# 可視化
plt.figure(figsize=(12, 12))  # サイズ調整
for i, feature in enumerate(features_group1):
    plt.subplot(len(features_group1), 1, i + 1)
    sns.lineplot(x='play_number', y=feature, data=df)
    plt.title(f'Trend in {feature} over Play Number')
    plt.tight_layout()

# トレンドの統計的検出
for feature in features_group1:
    X = sm.add_constant(df['play_number'])  # 定数項を追加
    y = df[feature]
    model = sm.OLS(y, X).fit()
    print(f'Trend analysis for {feature}:')
    print('Parameters:', model.params)
    print('P-values:', model.pvalues)
    print('R-squared:', model.rsquared)
    print('------------------------------------------------')

# プロットの表示
plt.show()
# 第2グループの特徴量
features_group2 = ['key', 'speechiness', 'mode', 'valence', 'liveness', 'tempo_scaled']

# 可視化
plt.figure(figsize=(12, 12))  # サイズ調整
for i, feature in enumerate(features_group2):
    plt.subplot(len(features_group2), 1, i + 1)
    sns.lineplot(x='play_number', y=feature, data=df)
    plt.title(f'Trend in {feature} over Play Number')
    plt.tight_layout()

# トレンドの統計的検出
for feature in features_group2:
    X = sm.add_constant(df['play_number'])  # 定数項を追加
    y = df[feature]
    model = sm.OLS(y, X).fit()
    print(f'Trend analysis for {feature}:')
    print('Parameters:', model.params)
    print('P-values:', model.pvalues)
    print('R-squared:', model.rsquared)
    print('------------------------------------------------')

# プロットの表示
plt.show()

