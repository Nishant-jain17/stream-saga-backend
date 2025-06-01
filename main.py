from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:3006",
        "https://stream-saga-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JIKAN_API_BASE = 'https://api.jikan.moe/v4'

class RecommendationRequest(BaseModel):
    content_type: str 
    titles: List[str]
    preferred_genres: List[int]

def sanitize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    else:
        return obj

def fetch_data_by_title(title: str, content_type: str) -> dict:
    url = f'{JIKAN_API_BASE}/{content_type}?q={title}&limit=1'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get('data', [None])[0]

def fetch_similar_by_genres(genre_ids: List[int], content_type: str, num_items: int = 20) -> List[dict]:
    genre_str = ','.join(map(str, genre_ids))
    url = f'{JIKAN_API_BASE}/{content_type}?genres={genre_str}&limit={num_items}'
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get('data', [])

def build_content_filtering(content_data: List[dict]):
    df = pd.DataFrame(content_data)
    if 'genre_ids' in df.columns:
        df['genre_str'] = df['genre_ids'].apply(lambda ids: ' '.join(map(str, ids)) if isinstance(ids, list) else '')
    else:
        df['genre_str'] = ''
    if df['genre_str'].str.strip().eq('').all():
        return df, None
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(df['genre_str'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return df, cosine_sim

def get_recommendations(df, cosine_sim, idx=0, num_recommendations=5, exclude_ids=[]):
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in sim_scores[1:]:
        content_id = df.iloc[i]['id']
        if content_id not in exclude_ids:
            recommendations.append({
                'id': int(df.iloc[i]['id']),
                'title': df.iloc[i]['title'],
                'genre_ids': df.iloc[i]['genre_ids'],
                'overview': df.iloc[i].get('overview', ''),
                'release_date': df.iloc[i].get('release_date'),
                'rating': df.iloc[i].get('rating'),
                'popularity': df.iloc[i].get('popularity'),
                'url': df.iloc[i].get('url'),
                'image_url': df.iloc[i].get('image_url')
            })
            if len(recommendations) >= num_recommendations:
                break
    return recommendations

@app.post("/recommendations")
def recommend(req: RecommendationRequest):
    content_type = req.content_type.lower()
    titles = req.titles
    preferred_genres = req.preferred_genres

    if not titles:
        raise HTTPException(status_code=400, detail="No titles provided")
    if content_type not in ['anime', 'manga']:
        raise HTTPException(status_code=400, detail="Unsupported content type")

    content_data = []
    user_ids = []

    # Fetch detailed info for each user title
    for title in titles:
        info = fetch_data_by_title(title, content_type)
        if info:
            user_ids.append(info['mal_id'])
            release_date_raw = info.get('aired', {}).get('from') if content_type == 'anime' else info.get('published', {}).get('from')
            release_date = release_date_raw.split("T")[0] if release_date_raw else None
            content_data.append({
                'id': info['mal_id'],
                'title': info['title'],
                'genre_ids': [g['mal_id'] for g in info.get('genres', [])],
                'overview': info.get('synopsis', ''),
                'release_date': release_date,
                'rating': info.get('score'),
                'popularity': info.get('popularity'),
                'url': info.get('url'),
                'image_url': info.get('images', {}).get('jpg', {}).get('image_url') if info.get('images') else None
            })

    # Fetch additional content by genres to enlarge dataset
    additional = fetch_similar_by_genres(preferred_genres, content_type)

    for item in additional:
        item_id = item.get('mal_id')
        if item_id and item_id not in user_ids:
            release_date_raw = item.get('aired', {}).get('from') if content_type == 'anime' else item.get('published', {}).get('from')
            release_date = release_date_raw.split("T")[0] if release_date_raw else None
            content_data.append({
                'id': item_id,
                'title': item.get('title'),
                'genre_ids': [g['mal_id'] for g in item.get('genres', [])] if item.get('genres') else [],
                'overview': item.get('synopsis', ''),
                'release_date': release_date,
                'rating': item.get('score'),
                'popularity': item.get('popularity'),
                'url': item.get('url'),
                'image_url': item.get('images', {}).get('jpg', {}).get('image_url') if item.get('images') else None
            })

    if not content_data:
        raise HTTPException(status_code=404, detail="No content data found")

    df, cosine_sim = build_content_filtering(content_data)

    if df.empty or cosine_sim is None:
        raise HTTPException(status_code=404, detail="Not enough data to generate recommendations")

    recommendations = get_recommendations(df, cosine_sim, 0, 5, user_ids)

    return {"recommendations": sanitize(recommendations)}
