import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------------
# 기본 설정 및 제목
# ---------------------
st.set_page_config(page_title="애니 추천 웹앱", layout="wide")
st.title("🎌 애니메이션 추천 웹앱")
st.markdown(
    "[📂 데이터 출처 (Kaggle)](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=anime.csv)",
    unsafe_allow_html=True,
)

# ---------------------
# 데이터 불러오기 및 전처리
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name"])  # 이름 없는 행 제거
    df["genre"] = df["genre"].fillna("").str.lower()
    df["type"] = df["type"].fillna("Unknown")

    # Gintama 통합
    gintama_mask = df["name"].str.contains("Gintama", case=False, na=False)
    if gintama_mask.sum() > 1:
        gintama_entry = df[gintama_mask].sort_values("members", ascending=False).iloc[0]
        df = df[~gintama_mask]
        df = pd.concat([df, pd.DataFrame([gintama_entry])], ignore_index=True)

    return df

anime_df = load_data()

# ---------------------
# 사용자 입력: 장르, 형식, 평점/인기도 범위
# ---------------------
with st.sidebar:
    st.header("🔍 필터 옵션")

    all_genres = sorted(set(g for genre in anime_df["genre"].dropna() for g in genre.split(", ")))
    selected_genres = st.multiselect("장르 선택 (다중 선택 가능)", options=all_genres)

    all_types = sorted(anime_df["type"].dropna().unique())
    selected_types = st.multiselect("형식 선택 (다중 선택 가능)", options=all_types)

    min_score, max_score = st.slider("평점 범위", 0.0, 10.0, (5.0, 10.0), step=0.1)
    min_members, max_members = st.slider("인기도 범위 (members 수)", 0, int(anime_df["members"].max()), (1000, 500000))

# ---------------------
# 필터링 적용
# ---------------------
filtered_df = anime_df.copy()

if selected_genres:
    filtered_df = filtered_df[filtered_df["genre"].apply(lambda g: any(gen in g for gen in selected_genres))]

if selected_types:
    filtered_df = filtered_df[filtered_df["type"].isin(selected_types)]

filtered_df = filtered_df[
    (filtered_df["rating"] >= min_score) &
    (filtered_df["members"] >= min_members) &
    (filtered_df["members"] <= max_members)
]

# ---------------------
# 시각화: 평점 & 인기도 (가로 막대)
# ---------------------
st.subheader("📊 평점 기준 상위 애니메이션")
top_rating = filtered_df.sort_values("rating", ascending=False).head(10)
fig_rating = px.bar(top_rating, x="rating", y="name", orientation="h", color="type", title="Top 10 by Rating")
st.plotly_chart(fig_rating, use_container_width=True)

st.subheader("📈 인기도 기준 상위 애니메이션")
top_members = filtered_df.sort_values("members", ascending=False).head(10)
fig_members = px.bar(top_members, x="members", y="name", orientation="h", color="type", title="Top 10 by Popularity")
st.plotly_chart(fig_members, use_container_width=True)

# ---------------------
# 추천 모드 선택
# ---------------------
st.header("🎯 애니메이션 추천")
recommend_mode = st.radio("추천 방식 선택", ["선택한 필터 기반", "입력한 애니 기반"])

# ---------------------
# 이미지 제외 장르
# ---------------------
EXCLUDED_GENRES_FOR_IMAGE = {"hentai", "ecchi", "horror", "yaoi"}

def get_anime_image(title, genre=""):
    if any(bad in genre.lower() for bad in EXCLUDED_GENRES_FOR_IMAGE):
        return None

    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        resp = requests.get(url)
        data = resp.json()
        return data["data"][0]["images"]["jpg"]["image_url"]
    except:
        return None

def generate_wordcloud(text):
    if not text:
        return None
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    buf = BytesIO()
    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    return buf

# ---------------------
# 1. 필터 기반 추천
# ---------------------
if recommend_mode == "선택한 필터 기반":
    st.subheader("🔎 필터 기반 추천 결과")
    if filtered_df.empty:
        st.warning("조건에 맞는 애니메이션이 없습니다.")
    else:
        for _, row in filtered_df.sort_values("rating", ascending=False).head(10).iterrows():
            col1, col2 = st.columns([1, 2])
            with col1:
                img_url = get_anime_image(row["name"], row["genre"])
                if img_url:
                    st.image(img_url, caption=row["name"])
                else:
                    st.image("https://via.placeholder.com/150?text=No+Image", caption=row["name"])
            with col2:
                if synopsis and not genre_set.intersection(EXCLUDED_IMAGE_GENRES):
                    wc_buf = generate_wordcloud(synopsis)
                    st.image(wc_buf, caption="📚 워드클라우드 (시놉시스 기반)", use_container_width=True)
                else:
                    st.write("워드클라우드 없음")


# ---------------------
# 2. 입력 기반 추천 (Content-based)
# ---------------------
else:
    st.subheader("🧠 입력한 애니와 유사한 애니 추천")

    anime_options = anime_df["name"].sort_values().unique().tolist()
    selected_titles = st.multiselect("기준이 될 애니메이션 선택", anime_options)

    def recommend_by_content(df, selected_titles):
        df = df.dropna(subset=["genre"])
        df = df.reset_index(drop=True)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["genre"])

        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index, index=df["name"]).drop_duplicates()

        selected_indices = indices[selected_titles].values
        sim_scores = sim_matrix[selected_indices].mean(axis=0)

        df["similarity"] = sim_scores
        recommended = df[~df["name"].isin(selected_titles)]
        return recommended.sort_values("similarity", ascending=False).head(10)

    if selected_titles:
        results = recommend_by_content(anime_df, selected_titles)
        for _, row in results.iterrows():
            col1, col2 = st.columns([1, 2])
            with col1:
                img_url = get_anime_image(row["name"], row["genre"])
                if img_url:
                    st.image(img_url, caption=row["name"])
                else:
                    st.image("https://via.placeholder.com/150?text=No+Image", caption=row["name"])
            with col2:
                st.markdown(f"**{row['name']}**  \n⭐ 평점: {row['rating']}  \n👥 인기도: {row['members']}  \n🎞️ 형식: {row['type']}")
                wc_buf = generate_wordcloud(row["genre"])
                if wc_buf:
                    st.image(wc_buf, caption="📌 장르 WordCloud", use_container_width=True)
    else:
        st.info("왼쪽에서 기준 애니메이션을 선택해주세요.")
