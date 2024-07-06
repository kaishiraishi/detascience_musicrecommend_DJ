import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # 特徴量とID、名前を分離
    features = data.drop(columns=['playlist_number', 'duration_ms', 'tempo'])
    return features

def initialize_playlist(data, num_tracks=10):
    # 初期プレイリストをランダムに選択
    initial_indices = np.random.choice(data.index, size=num_tracks, replace=False)
    return data.iloc[initial_indices]

def predict_next_track(model, current_playlist, all_features):
    # 特徴量名をリストで指定
    feature_columns = [
        'danceability', 'acousticness', 'energy', 'instrumentalness',
        'loudness', 'liveness', 'key', 'valence', 'speechiness', 'mode','play_number', 'tempo_scaled'
    ]

    # 現在のプレイリストから最後の10曲の特徴量を取得
    current_features = current_playlist[-10:][feature_columns].values

    # モデルによる次の曲の特徴量を予測
    prediction = model.predict(current_features[np.newaxis, :, :])[0]

    # 全曲の特徴量とのコサイン類似度を計算（全特徴量データセットから特定の特徴量カラムを選択する）
    similarity_scores = cosine_similarity([prediction], all_features[feature_columns].values)[0]

    # 最も類似度が高い曲のインデックスを取得
    next_track_index = np.argmax(similarity_scores)
    return all_features.iloc[next_track_index]


def generate_playlist(model, initial_playlist, all_features, playlist_length=50):
    current_playlist = initial_playlist.copy()
    used_indices = set(initial_playlist.index)  # すでに選ばれた曲のインデックスを保存

    for _ in range(playlist_length - len(initial_playlist)):
        next_track = predict_next_track(model, current_playlist, all_features)
        
        # 選ばれた曲が既にプレイリストにある場合は別の曲を選ぶ
        while next_track.name in used_indices:
            next_track = predict_next_track(model, current_playlist, all_features)

        used_indices.add(next_track.name)  # 新しく選ばれた曲のインデックスを追加
        current_playlist = pd.concat([current_playlist, pd.DataFrame([next_track])], ignore_index=True)

    return current_playlist



if __name__ == '__main__':
    model_path = './model/1rnn_model.h5'
    data_path = './data/ALL_track_features_updated.csv'

    # モデルとデータの読み込み
    model = load_model(model_path)
    data = load_data(data_path)
    all_features = preprocess_data(data)

    # 初期プレイリストの作成
    initial_playlist = initialize_playlist(data, num_tracks=10)

    # プレイリストの生成
    final_playlist = generate_playlist(model, initial_playlist, all_features, playlist_length=50)

    # 生成されたプレイリストをCSVファイルに保存
    final_playlist[['track_id', 'track_name']].to_csv('generated_playlist.csv', index=False)
