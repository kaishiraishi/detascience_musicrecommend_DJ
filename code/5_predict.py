import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    features = data.drop(columns=['track_id', 'track_name'])  # トラックIDと名前を除外
    return features.values  # Numpy配列に変換

def initialize_playlist(data, num_tracks=10):
    # 初期プレイリストをランダムに選択
    initial_indices = np.random.choice(range(len(data)), size=num_tracks, replace=False)
    return data[initial_indices]

def predict_next_track(model, current_playlist, all_features):
    # 最後の10曲から次の曲を予測
    prediction = model.predict(np.array([current_playlist[-10:]]))[0]
    # 全曲とのコサイン類似度を計算
    similarity_scores = cosine_similarity([prediction], all_features)[0]
    # 最も類似度が高い曲のインデックスを取得
    next_track_index = np.argmax(similarity_scores)
    return all_features[next_track_index]

def generate_playlist(model, initial_playlist, all_features, playlist_length=50):
    current_playlist = list(initial_playlist)
    for _ in range(playlist_length - len(initial_playlist)):
        next_track = predict_next_track(model, np.array(current_playlist), all_features)
        current_playlist.append(next_track)
    return np.array(current_playlist)

# メイン実行ブロック
if __name__ == '__main__':
    model_path = './model/rnn_model.h5'
    data_path = './data/ALL_track_features_updated.csv'

    # モデルとデータの読み込み
    model = load_model(model_path)
    data = load_data(data_path)
    all_features = preprocess_data(data)

    # 初期プレイリストの作成
    initial_playlist = initialize_playlist(all_features, num_tracks=10)

    # プレイリストの生成
    final_playlist = generate_playlist(model, initial_playlist, all_features, playlist_length=50)

    # 生成されたプレイリストの保存
    np.savetxt('generated_playlist.csv', final_playlist, delimiter=',')
