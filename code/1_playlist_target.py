import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm

# Spotify APIの設定
client_id = '264f764417e24da8abe1508c468df6c0'
client_secret = '4626a88350f64bb1a0d7d7a9577342c0'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_all_track_ids_and_position(playlist_id, playlist_number):
    track_details = []
    results = sp.playlist_tracks(playlist_id)
    position = 1

    for item in tqdm(results['items'], desc="Processing tracks"):
        track_id = item['track']['id']
        if track_id:
            track_details.append((track_id, position, 0))
        position += 1

    while results['next']:
        time.sleep(1)  # APIコール間に1秒の遅延を設ける
        results = sp.next(results)
        for item in tqdm(results['items'], desc="Processing tracks"):
            track_id = item['track']['id']
            if track_id:
                track_details.append((track_id, position, 0))
            position += 1

    return track_details

def get_track_features(track_details):
    features_list = []
    batch_size = 50  # 一度に処理するトラックの数
    for i in tqdm(range(0, len(track_details), batch_size), desc="Fetching track features"):
        batch = track_details[i:i+batch_size]
        track_ids = [track[0] for track in batch]
        try:
            tracks_info = sp.tracks(track_ids)['tracks']
            tracks_features = sp.audio_features(track_ids)
            for index, (info, features) in enumerate(zip(tracks_info, tracks_features)):
                if features:
                    features_list.append({
                        'track_id': info['id'],
                        'track_name': info['name'],
                        'danceability': features['danceability'],
                        'acousticness': features['acousticness'],
                        'tempo': features['tempo'],
                        'instrumentalness': features['instrumentalness'],
                        'energy': features['energy'],
                        'key': features['key'],
                        'speechiness': features['speechiness'],
                        'mode': features['mode'],
                        'valence': features['valence'],
                        'liveness': features['liveness'],
                        'play_number': batch[index][1],
                        'playlist_number': batch[index][2]
                    })
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers['Retry-After'])
                print(f"Rate limit exceeded, sleeping for {retry_after} seconds.")
                time.sleep(retry_after + 1)  # レートリミット超過時に指定された時間待機
                continue
            else:
                print(f"An error occurred: {e}")
                break

    return features_list

#######################################################################################################
#ALL
#playlist_ids = ['54PDYvBhOWsLQdNx0QBfir','5zpoWFQ07vZQgxI2Te9QN0']

#setlist
playlist_ids = ["7q1v2XEtG7VKL43a3KRDZS","2x9AJ9CMCWuy3z6Z60FwzB","3jh079OCXqZyjO5lIIbxz0","2HJiPFLnU8evA9LqFHw6E8","0jNyIKL77YOoss7kgSkcKu","7jvvWCY9lV742jA7WlTnAa","3gayI3Ix4VJH8XaccbdFzV","4bDTpvyCgukzBrcXQAkd1Y","5nSHOShqekXxFMkZxGnUDL","7nKCHh113ojicmJ8QMCvhF"]

track_details = []

for index, playlist_id in enumerate(playlist_ids, start=1):
    details = get_all_track_ids_and_position(playlist_id, index)
    track_details.extend(details)

track_features = get_track_features(track_details)
df = pd.DataFrame(track_features)

# 重複する行を削除（トラックIDとトラック名で検討）
df = df.drop_duplicates(subset=['track_id', 'track_name'])


######################################################################################################
# CSVファイルに保存
df.to_csv('./data/target.csv', index=False)

print("CSV file has been created with the track features.")
