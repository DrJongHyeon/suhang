# anime_recommender_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Constants
DEFAULT_IMG_URL = "https://via.placeholder.com/150"
EXCLUDED_IMAGE_GENRES = {"Hentai", "Ecchi", "Horror", "Yaoi"}

@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df.dropna(subset=["genre", "rating", "type"], inplace=True)
    df = df[df["episodes"].astype(str).str.isdigit()]
    df["episodes"] = df["episodes"].astype(int)
    df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in x.split(",")])
    df["series_name"] = df["name"].str.extract(r"(^[^:!\(\)]*)")
    df = df.sort_values("members", ascending=False).drop_duplicates("series_name")
    return df.reset_index(drop=True)

def filter_anime(df, genres, types, rating_range, member_range, keyword):
    filtered = df.copy()
    if genres:
        filtered = filtered[filtered["genre_list"].apply(lambda g: set(genres).issubset(g))]
    if types:
        filtered = filtered[filtered["type"].isin(types)]
    if keyword:
        filtered = filtered[filtered["name"].str.contains(keyword, case=False)]
    filtered = filtered[(filtered["rating"] >= rating_range[0]) & (filtered["rating"] <= rating_range[1])]
    filtered = filtered[(filtered["members"] >= member_range[0]) & (filtered["members"] <= member_range[1])]
    return filtered

def get_anime_info(title):
    try:
        res = requests.get(f"https://api.jikan.moe/v4/anime", params={"q": title, "limit": 1}).json()
        if res.get("data"):
            anime = res["data"][0]
            return anime.get("images", {}).get("jpg", {}).get("image_url", DEFAULT_IMG_URL), anime.get("synopsis", "")
    except:
        pass
    return DEFAULT_IMG_URL, ""

def generate_wordcloud(text):
    wc = WordCloud(width=400, height=300, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def build_similarity_model(df):
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df["genre_list"])
    df_features = pd.DataFrame(genre_encoded, index=df.index)
    df_features["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df_features["members"] = pd.to_numeric(df["members"], errors="coerce")
    df_features["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df_features.dropna(inplace=True)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df_features.to_numpy())  # 🔧 핵심 수정 부분
    return features_scaled, df_features.index

def recommend_similar(df, features_scaled, valid_indices, selected_titles, top_n=10):
    df = df.loc[valid_indices].copy()
    indices = df[df["name"].isin(selected_titles)].index
    if not len(indices):
        return pd.DataFrame()
    selected_vec = features_scaled[df.index.get_indexer(indices)].mean(axis=0).reshape(1, -1)
    sim_scores = cosine_similarity(selected_vec, features_scaled).flatten()
    df["similarity"] = sim_scores
    recs = df[~df["name"].isin(selected_titles)].sort_values("similarity", ascending=False).head(top_n)
    return recs

# Load data
anime_df = load_data()
all_genres = sorted(set(g for sub in anime_df["genre_list"] for g in sub))
all_types = sorted(anime_df["type"].dropna().unique())
features_scaled, valid_indices = build_similarity_model(anime_df)

# UI
st.title("🎌 Anime Recommender")
mode = st.radio("Select Recommendation Mode", ["🎯 조건 기반 추천", "🤖 유사 애니 추천"])

if mode == "🎯 조건 기반 추천":
    genres = st.multiselect("장르 선택", all_genres)
    types = st.multiselect("형식 선택", all_types)
    rating_range = st.slider("평점 범위", 0.0, 10.0, (7.0, 10.0), step=0.1)
    member_range = st.slider("인기도 (members) 범위", int(anime_df["members"].min()), int(anime_df["members"].max()), (10000, 500000))
    keyword = st.text_input("제목 키워드 검색")

    filtered_df = filter_anime(anime_df, genres, types, rating_range, member_range, keyword)
    top_recommendations = filtered_df.sort_values(by="rating", ascending=False).head(10)

else:
    selected_titles = st.multiselect("좋아하는 애니메이션을 선택하세요", anime_df["name"].tolist())
    top_recommendations = recommend_similar(anime_df.copy(), features_scaled, valid_indices, selected_titles)

# 출력
st.subheader("📌 추천 애니메이션")
if top_recommendations.empty:
    st.info("조건에 맞는 추천 결과가 없습니다.")
else:
    for _, row in top_recommendations.iterrows():
        if EXCLUDED_IMAGE_GENRES.intersection(set(row["genre_list"])):
            continue
        img_url, synopsis = get_anime_info(row["name"])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_url or DEFAULT_IMG_URL, caption=row["name"], use_container_width=True)
        with col2:
            st.markdown(f"**평점:** {row['rating']} | **인기도:** {row['members']:,}")
            st.markdown(synopsis if synopsis else "(시놉시스 정보 없음)")
            if synopsis:
                generate_wordcloud(synopsis)

    st.subheader("📊 평점순 Top 10")
    fig1 = px.bar(top_recommendations.sort_values("rating"), x="rating", y="name", orientation="h", color="type")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 인기도순 Top 10")
    fig2 = px.bar(top_recommendations.sort_values("members"), x="members", y="name", orientation="h", color="type")
    st.plotly_chart(fig2, use_container_width=True)
