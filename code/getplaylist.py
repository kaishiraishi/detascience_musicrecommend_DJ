import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import time

# Spotify APIの設定
client_id = '264f764417e24da8abe1508c468df6c0'
client_secret = '4626a88350f64bb1a0d7d7a9577342c0'
redirect_uri = 'http://localhost:8888/callback'
scope = 'user-library-read playlist-modify-public'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))

def get_playlist_tracks(sp, playlist_id):
    """既存のプレイリストからすべてのトラックIDを取得する関数"""
    tracks = []
    response = sp.playlist_tracks(playlist_id)
    while response:
        tracks.extend([item['track']['id'] for item in response['items']])
        response = sp.next(response) if response['next'] else None
    return set(tracks)

def safe_spotify_call(call):
    """APIコールを安全に行い、必要に応じてレートリミットに対処する関数"""
    try:
        return call()
    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 429:
            retry_after = int(e.headers['Retry-After'])
            time.sleep(retry_after + 1)
            return call()
        else:
            raise

def get_recommendations_and_add_to_playlist(playlist_id, number_of_recommendations_per_track):
    existing_track_ids = get_playlist_tracks(sp, playlist_id)
    tracks = sp.playlist_tracks(playlist_id)
    new_track_ids = []

    for item in tqdm(tracks['items'], desc="Getting recommendations"):
        seed_track_id = item['track']['id']
        if seed_track_id:
            recommendations = safe_spotify_call(lambda: sp.recommendations(seed_tracks=[seed_track_id], limit=number_of_recommendations_per_track))
            for track in recommendations['tracks']:
                if track['id'] not in existing_track_ids and track['id'] not in new_track_ids:
                    new_track_ids.append(track['id'])

    # トラックの追加を100曲ずつに分けて行う
    for i in range(0, len(new_track_ids), 100):
        batch_tracks = new_track_ids[i:i + 100]
        safe_spotify_call(lambda: sp.user_playlist_add_tracks(user=sp.me()['id'], playlist_id=playlist_id, tracks=batch_tracks))
        print(f"Added {len(batch_tracks)} tracks to the playlist in batch.")

# プレイリストIDと1曲あたりの推薦曲数を指定
playlist_id = '6bGVhIBpTf6Kl7hPisuaY9'
number_of_recommendations_per_track = 5

get_recommendations_and_add_to_playlist(playlist_id, number_of_recommendations_per_track)
