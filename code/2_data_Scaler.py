from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv('./data/target.csv')
# 'ALL_track_features.csv'ファイルの読み込み
df_all = pd.read_csv('./data/all.csv')



# MinMaxScalerのインスタンス作成
scaler = MinMaxScaler()
# MinMaxScalerのインスタンス作成
scaler_all = MinMaxScaler()



# 'tempo'のデータを0-1の範囲でスケーリング
df['tempo_scaled'] = scaler.fit_transform(df[['tempo']])
# 'tempo'のデータを0-1の範囲でスケーリング
df_all['tempo_scaled'] = scaler_all.fit_transform(df_all[['tempo']])



# スケーリング後の値を含む最初の5行を表示
print(df[['tempo', 'tempo_scaled']].head())
# スケーリング後の値を含む最初の5行を表示
print(df_all[['tempo', 'tempo_scaled']].head())



#################################################################
#################################################################
#################################################################


# 変更を新しいファイルに保存
df.to_csv('target_scaled.csv', index=False)
# 変更を新しいファイルに保存
df_all.to_csv('all_scaled.csv', index=False)

