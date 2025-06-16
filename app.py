import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- 기본 설정 ---
st.set_page_config(page_title="애니 추천 웹앱", layout="wide")
st.title("🎌 애니메이션 추천 웹앱")

# --- 데이터 로딩 및 전처리 ---
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")

    # Gintama 시리즈 통합 (members 가장 높은 하나로 대체)
    gintama_mask = df["name"].str.contains("Gintama", case=False, na=False)
    if gintama_mask.any():
        gintama_group = df[gintama_mask].sort_values("members", ascending=False).iloc[0]
        df = df[~gintama_mask]
        df = pd.concat([df, pd.DataFrame([gintama_group])], ignore_index=True)

    df["genre"] = df["genre"].fillna("Unknown")
    df["genre_list"] = df["genre"].str.split(", ").apply(lambda x: [i.strip() for i in x])
    df["type"] = df["type"].fillna("Unknown")

    return df

anime_df = load_data()

# --- 사용자 입력 받기 ---
genre_options = sorted(set(g for sublist in anime_df["genre_list"] for g in sublist))
type_options = sorted(anime_df["type"].unique())

selected_genres = st.multiselect("🎭 장르 선택 (복수 선택 가능)", genre_options)
selected_types = st.multiselect("📺 형식 선택 (복수 선택 가능)", type_options)

min_rating, max_rating = st.slider("⭐ 평점 범위 선택", 0.0, 10.0, (6.0, 10.0), 0.1)
min_members, max_members = st.slider("👥 인기도 범위 (Members 수)", 0, int(anime_df["members"].max()), (10000, 500000), step=10000)

# --- 필터링 ---
filtered_df = anime_df.copy()

if selected_genres:
    filtered_df = filtered_df[filtered_df["genre_list"].apply(lambda genres: any(g in genres for g in selected_genres))]

if selected_types:
    filtered_df = filtered_df[filtered_df["type"].isin(selected_types)]

filtered_df = filtered_df[
    (filtered_df["rating"].between(min_rating, max_rating)) &
    (filtered_df["members"].between(min_members, max_members))
]

# --- 결과 출력 ---
st.subheader("🎯 추천 애니메이션 Top 10")

top_recommendations = filtered_df.sort_values(by="rating", ascending=False).head(10)

# --- 이미지 출력 (Jikan API 사용) ---
def get_image_url(title, genre_list):
    # 제외할 장르 포함 시 이미지 출력 안 함
    excluded_genres = {"Hentai", "Ecchi", "Horror", "Yaoi"}
    if any(genre in excluded_genres for genre in genre_list):
        return "https://via.placeholder.com/150?text=No+Image"

    try:
        response = requests.get(f"https://api.jikan.moe/v4/anime", params={"q": title, "limit": 1})
        data = response.json()
        return data["data"][0]["images"]["jpg"]["image_url"]
    except:
        return "https://via.placeholder.com/150?text=No+Image"

# --- 워드클라우드 생성 ---
def generate_wordcloud(text):
    wordcloud = WordCloud(width=400, height=300, background_color="white").generate(text)
    return wordcloud

# --- 추천 결과 시각화 & 출력 ---
for _, row in top_recommendations.iterrows():
    col1, col2 = st.columns([1, 3])
    with col1:
        img_url = get_image_url(row["name"], row["genre_list"])
        st.image(img_url, caption=row["name"], use_container_width=True)

    with col2:
        # 이미지 제외 조건 장르 아니면 워드클라우드 생성
        if not any(g in {"Hentai", "Ecchi", "Horror", "Yaoi"} for g in row["genre_list"]):
            st.markdown(f"**{row['name']}** - 평점: {row['rating']}, 인기도: {row['members']}")
            wc = generate_wordcloud(" ".join(row["genre_list"]))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

st.markdown("---")

# --- 시각화: 평점/인기도 상위 막대 그래프 (가로) ---
st.subheader("📊 평점 & 인기도 시각화")

if not filtered_df.empty:
    top_rated = filtered_df.sort_values(by="rating", ascending=False).head(10)
    top_members = filtered_df.sort_values(by="members", ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(top_rated, x="rating", y="name", orientation="h", title="Top 10 평점 높은 애니")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(top_members, x="members", y="name", orientation="h", title="Top 10 인기 애니 (Members 기준)")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("조건에 맞는 애니메이션이 없습니다.")

# --- 콘텐츠 기반 추천 기능 (선택한 애니 기준) ---
st.markdown("---")
st.subheader("💡 콘텐츠 기반 애니 추천 (선택한 애니 기준)")

anime_choices = anime_df["name"].unique()
selected_titles = st.multiselect("추천 기준으로 삼을 애니메이션 선택", anime_choices)

def build_similarity_model(df):
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df["genre_list"])
    df_features = pd.DataFrame(genre_encoded, index=df.index)
    df_features["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df_features["members"] = pd.to_numeric(df["members"], errors="coerce")
    df_features["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df_features.dropna(inplace=True)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df_features.to_numpy())  # 핵심 수정

    return features_scaled, df_features.index

def recommend_by_content(selected_titles, df):
    features_scaled, valid_indices = build_similarity_model(df)
    df_valid = df.loc[valid_indices].reset_index(drop=True)

    selected_indices = df_valid[df_valid["name"].isin(selected_titles)].index
    if len(selected_indices) == 0:
        return pd.DataFrame()

    sim_matrix = cosine_similarity(features_scaled)
    sim_scores = sim_matrix[selected_indices].mean(axis=0)

    df_valid["similarity"] = sim_scores
    recommended = df_valid[~df_valid["name"].isin(selected_titles)].sort_values(by="similarity", ascending=False)
    return recommended.head(10)

if selected_titles:
    recs = recommend_by_content(selected_titles, anime_df)
    st.markdown("#### 🔁 유사한 애니 추천:")
    for _, row in recs.iterrows():
        st.markdown(f"- {row['name']} (유사도: {row['similarity']:.2f})")
