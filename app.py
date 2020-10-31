
import sqlite3
import numpy as np
import pandas as pd
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
import time

def connect_to_database(PATH1, attach=None):
    if os.path.isfile(PATH1):
        conn = sqlite3.connect(PATH1, check_same_thread=False)
        cursor = conn.cursor()
    else:
        print("Database file does not exists")

    if attach:
        if os.path.isfile(attach):
            cursor.execute(f"ATTACH '{attach}' AS meta")
        else:
            print("Attach database file note found")

    return cursor

cursor = connect_to_database('mxm_dataset.db', 'track_metadata.db')

def load_json(PATH):
    if os.path.isfile(PATH):
        with open(PATH, 'r') as f:
            data = json.load(f)
    else:
        print(f"{PATH} is not a file")

    return data

# Load data into files or mmaps
word_mappings = load_json('word_mappings')
track_lyric_mappings = load_json('track_lyric_mapping')
tf_idf_matrix = np.load('tf_idf_matrix.npy', mmap_mode='r')
# Set parameters for number of songs and columns
n_rows = tf_idf_matrix.shape[0]
n_cols = tf_idf_matrix.shape[1]

# Create track_id mapper
def calc_10_most_similar(song_id, tf_idf_matrix, n=10):
    """Returns the index of the 10 most similar songs as well as a cosine similarity score"""
    numerator = np.dot(tf_idf_matrix, tf_idf_matrix[song_id,:].reshape(-1,1))
    denominator = np.linalg.norm(tf_idf_matrix[song_id]) * np.linalg.norm(tf_idf_matrix, axis=1).reshape(-1,1)
    cosine_sim = (numerator / denominator).reshape(-1)
    sort_indices = np.argsort(-cosine_sim)
    return sort_indices[:n], cosine_sim

def extract_most_similar_names(song_id, tf_idf_matrix,n=10):
    """From a given song ID, returns the top n songs (artist and track) as well as cosine similarity"""
    sort_indices, cosine_sim = calc_10_most_similar(song_id, tf_idf_matrix, n=n)

    # Return the track_ids of the top n most similar along with cosine similarity
    top_n = [track_lyric_mappings[str(x)] for x in sort_indices]
    cosine_sim = [x for x in cosine_sim[sort_indices]]

    songs = []
    for idx, song in enumerate(top_n):
        query_str = (song,)
        cursor.execute("SELECT title, artist_name FROM lyric_metadata WHERE track_id = ?", query_str)
        song = cursor.fetchall()
        songs.append((*song,cosine_sim[idx]))

    return songs

def track_id_to_index(track_id, track_lyric_mappings):
    for key, value in track_lyric_mappings.items():
        if value == track_id:
            return int(key)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H1(
            "Sentiment Analysis"
        )
    ], style={'textAlign':'center'}),
    html.Div([
        dcc.Input(
            id='artist-text', 
            type='text',
            value='Artist',
        ),
        html.Button(
            id='submit-button-state', 
            n_clicks=0, 
            children='Submit'
        )
    ],  style = {'width': '100%',
                 'display': 'flex',
                 'align-items': 'center',
                 'justify-content': 'center'}
    ),
    html.Div([
        dcc.Dropdown(
            id='song-artist-options'
        )
    ], style={'margin':'50px 50px 75px 100px'}),
    html.Div([
        dcc.Loading(
            id='loading-cosine-sim',
            type='default',
            children=html.Div(id='output-state')
        ),
    ], style = {'width': '100%',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center'}
    )
])
@app.callback(
    Output('song-artist-options', 'options'),
    [Input('artist-text', 'value')])
def create_artist_query(user_query):
    query = f'SELECT artist_name, title FROM lyric_metadata WHERE artist_name LIKE "%{user_query}%"'
    cursor.execute(query)
    query_result = cursor.fetchall()
    query_result = [f"{x[0]} - {x[1]}" for x in query_result]
    return [{'label':i, 'value':i} for i in query_result]

@app.callback(
    Output('output-state', 'children'),
    [Input('song-artist-options', 'value')])
def test_func(selection):
    query = selection.split(" - ")
    artist, song = query
    track_id = cursor.execute(f'SELECT track_id FROM lyric_metadata WHERE artist_name LIKE "%{artist}%" AND title LIKE "%{song}%"')
    track_id = cursor.fetchall()
    track_index = track_id_to_index(track_id[0][0], track_lyric_mappings)
    track_array = tf_idf_matrix[track_index]
    similar_songs = extract_most_similar_names(track_index, tf_idf_matrix, n=10)
    formatted_songs = html.Div([
        html.H3("Songs with similar lyrical content"),
        html.Ul([html.Li(f"{x[0][0]} - {x[0][1]} - {np.round(x[1], 2)}") for x in similar_songs])
    ])
    return formatted_songs 

@app.callback(
    Output('loading-cosine-sim', 'children'),
    [Input('output-state', 'loading_state')])
def input_triggers_spinner(value):
    if value.is_loading:
        time.sleep(1)
        return value


if __name__ == '__main__':
    app.run_server(debug=True)


