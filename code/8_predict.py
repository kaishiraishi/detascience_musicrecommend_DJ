import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # 特徴量とID、名前を分離
    features = data.drop(columns=['playlist_number', 'tempo'])
    return features

def initialize_playlist(data, num_tracks=10):
    # 初期プレイリストをランダムに選択
    initial_indices = np.random.choice(data.index, size=num_tracks, replace=False)
    return data.iloc[initial_indices]



def predict_next_track(model, current_playlist, all_features):
    feature_columns = [
        'danceability', 'acousticness', 'instrumentalness','energy','key',  'speechiness', 'mode', 'valence','liveness','play_number', 'tempo_scaled'
    ]

    #track_id,track_name,danceability,acousticness,tempo,instrumentalness,energy,key,speechiness,mode,valence,liveness,play_number,playlist_number,tempo_scaled

    current_features = current_playlist[-10:][feature_columns].values
    prediction = model.predict(current_features[np.newaxis, :, :])[0]
    similarity_scores = cosine_similarity([prediction], all_features[feature_columns].values)[0]


###########################################################################
    # 類似度が0.8以上の曲を候補とする
    similar_tracks = all_features[similarity_scores >= 0.8]
    if not similar_tracks.empty:
        # ランダムに1曲選択
        return similar_tracks.sample(n=1).iloc[0]
    else:
        # 類似度が最も高い曲を選択
        next_track_index = np.argmax(similarity_scores)
        return all_features.iloc[next_track_index]






def generate_playlist(model, initial_playlist, all_features, playlist_length=50):
    current_playlist = initial_playlist.copy()
    used_indices = set(initial_playlist.index.tolist())  # すでに選ばれた曲のインデックスを保存

    for _ in tqdm(range(playlist_length - len(initial_playlist)), desc="Generating playlist"):
        next_track = predict_next_track(model, current_playlist, all_features)

        # next_track.name は、DataFrameから1行取得した場合のインデックスを返します
        while next_track.name in used_indices:
            next_track = predict_next_track(model, current_playlist, all_features)

        used_indices.add(next_track.name)  # 新しく選ばれた曲のインデックスを追加
        current_playlist = pd.concat([current_playlist, pd.DataFrame([next_track])], ignore_index=True)

    return current_playlist





if __name__ == '__main__':
    model_path = './model/rnn_model.h5'
    data_path = './data/C.csv'

    # モデルとデータの読み込み
    model = load_model(model_path)
    data = load_data(data_path)
    all_features = preprocess_data(data)

    # 初期プレイリストの作成
    initial_playlist = initialize_playlist(data, num_tracks=10)

    # プレイリストの生成
    final_playlist = generate_playlist(model, initial_playlist, all_features, playlist_length=50)

    # 生成されたプレイリストをCSVファイルに保存
    final_playlist[['track_id', 'track_name']].to_csv('./data/generated_playlist.csv', index=False)
