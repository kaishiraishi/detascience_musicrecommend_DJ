import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # 特徴量とID、名前を分離
    features = data.drop(columns=['play_number', 'playlist_number', 'duration_ms', 'tempo'])
    return features

def initialize_playlist(data, num_tracks=10):
    # 初期プレイリストをランダムに選択
    initial_indices = np.random.choice(data.index, size=num_tracks, replace=False)
    return data.iloc[initial_indices]

def predict_next_track(model, current_playlist, all_features):
    # 特徴量のみを取得
    current_features = current_playlist[-10:].drop(columns=['track_id', 'track_name']).values

    # モデルによる次の曲の特徴量を予測
    prediction = model.predict(current_features[np.newaxis, :, :])[0]

    # 全曲の特徴量とのコサイン類似度を計算
    similarity_scores = cosine_similarity([prediction], all_features.drop(columns=['track_id', 'track_name']).values)[0]

    # 最も類似度が高い曲のインデックスを取得
    next_track_index = np.argmax(similarity_scores)
    return all_features.iloc[next_track_index]

def generate_playlist(model, initial_playlist, all_features, playlist_length=50):
    current_playlist = initial_playlist.copy()
    for _ in range(playlist_length - len(initial_playlist)):
        next_track = predict_next_track(model, current_playlist, all_features)
        current_playlist = current_playlist.append(next_track, ignore_index=True)
    return current_playlist

if __name__ == '__main__':
    model_path = './model/rnn_model.h5'
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
