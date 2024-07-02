import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

client_id = '264f764417e24da8abe1508c468df6c0'
client_secret = '4626a88350f64bb1a0d7d7a9577342c0'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_all_track_ids_and_position(playlist_id, playlist_number):
    track_details = []  # トラックIDとその位置情報を保持するリスト
    results = sp.playlist_tracks(playlist_id)
    position = 1  # トラックの位置を追跡

    for item in results['items']:
        track_id = item['track']['id']
        if track_id:
            track_details.append((track_id, position, playlist_number))
        position += 1

    while results['next']:
        results = sp.next(results)
        for item in results['items']:
            track_id = item['track']['id']
            if track_id:
                track_details.append((track_id, position, playlist_number))
            position += 1

    return track_details

def get_track_features(track_details):
    features_list = []
    for track_id, play_number, playlist_number in track_details:
        try:
            track_info = sp.track(track_id)
            track_features = sp.audio_features(track_id)[0]

            features_list.append({
                'track_id': track_id,
                'track_name': track_info['name'],
                'danceability': track_features['danceability'],
                'acousticness': track_features['acousticness'],
                'energy': track_features['energy'],
                'tempo': track_features['tempo'],
                'instrumentalness': track_features['instrumentalness'],
                'loudness': track_features['loudness'],
                'liveness': track_features['liveness'],
                'duration_ms': track_features['duration_ms'],
                'key': track_features['key'],
                'valence': track_features['valence'],
                'speechiness': track_features['speechiness'],
                'mode': track_features['mode'],
                'play_number': play_number,
                'playlist_number': playlist_number
            })

        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers['Retry-After'])
                print(f"Rate limit exceeded, sleeping for {retry_after} seconds.")
                time.sleep(retry_after + 1)
                continue
            raise

    return features_list

# 複数のプレイリストIDをリストで定義  https://open.spotify.com/playlist/54PDYvBhOWsLQdNx0QBfir?si=77ce8fbb686f4231
playlist_ids = [
    '54PDYvBhOWsLQdNx0QBfir',

]



track_details = []

for index, playlist_id in enumerate(playlist_ids, start=1):
    details = get_all_track_ids_and_position(playlist_id, index)
    track_details.extend(details)

# トラックの特徴を取得
track_features = get_track_features(track_details)

# データフレームに結果を保存し、CSVに出力
df = pd.DataFrame(track_features)
df.to_csv('ALL_track_features.csv', index=False)

print("CSV file has been created with the track features.")
