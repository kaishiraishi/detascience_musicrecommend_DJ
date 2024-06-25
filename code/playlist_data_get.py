import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd


client_id = 'febed4a2364542c3b994eb7360478f3a'
client_secret = '4378e979300844bba06dfa6dfa1de147'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_all_track_ids(playlist_id):
    track_ids = []
    results = sp.playlist_tracks(playlist_id)
    

    for item in results['items']:
        track_id = item['track']['id']
        if track_id not in track_ids:
            track_ids.append(track_id)
    

    while results['next']:
        results = sp.next(results)
        for item in results['items']:
            track_id = item['track']['id']
            if track_id not in track_ids:
                track_ids.append(track_id)

    return track_ids

def get_track_features(track_ids):
    features_list = []
    

    for track_id in track_ids:
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
            'mode': track_features['mode']
        })
        
    return features_list

# 複数のプレイリストIDを処理
playlist_ids = ['3U3iLIRlGUBKVcfu8afnVG']
all_track_ids = []
for playlist_id in playlist_ids:
    track_ids = get_all_track_ids(playlist_id)
    all_track_ids.extend(track_id for track_id in track_ids if track_id not in all_track_ids)


track_features = get_track_features(all_track_ids)


df = pd.DataFrame(track_features)
df.to_csv('track_features.csv', index=False)

print("CSV file has been created with the track features.")