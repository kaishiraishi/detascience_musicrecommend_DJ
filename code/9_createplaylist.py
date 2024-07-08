import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from tqdm import tqdm

# Spotify API認証の設定
client_id = '264f764417e24da8abe1508c468df6c0'
client_secret = '4626a88350f64bb1a0d7d7a9577342c0'
redirect_uri = 'http://localhost:8888/callback'  # リダイレクトURI
scope = 'playlist-modify-public'  # 必要なスコープ

# 認証マネージャーを設定
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

def create_playlist_from_csv(playlist_name, csv_file):
    # CSVファイルからトラックのIDを読み込む
    df = pd.read_csv(csv_file)
    track_ids = df['track_id'].tolist()

    # 新しいプレイリストを作成（現在認証されているユーザーのもとに）
    playlist = sp.user_playlist_create(user=sp.me()['id'], name=playlist_name, public=True)
    playlist_id = playlist['id']

    # プレイリストにトラックを追加
    for i in tqdm(range(0, len(track_ids), 100)):
        batch = track_ids[i:i+100]
        sp.user_playlist_add_tracks(user=sp.me()['id'], playlist_id=playlist_id, tracks=batch)

    print(f"Playlist '{playlist_name}' has been created and filled with tracks from '{csv_file}'.")

playlist_name = 'ML'
csv_file = './data/generated_playlist.csv'

create_playlist_from_csv(playlist_name, csv_file)
