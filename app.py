import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re

# -------------------- 데이터 불러오기 --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["genre", "rating", "type", "name"])
    df = df[df["episodes"].apply(lambda x: x.isdigit())]
    df["episodes"] = df["episodes"].astype(int)
    df["genre"] = df["genre"].str.strip()
    df["genre_list"] = df["genre"].apply(lambda x: x.split(", "))
    
    # Gintama 시리즈 통합
    df["series_name"] = df["name"].apply(lambda x: "Gintama" if re.search(r"(?i)gintama", x) else x)

    # Gintama 중 가장 인기 있는 하나만 남기고 제거
    df = df.sort_values("members", ascending=False).drop_duplicates("series_name")

    return df

df = load_data()

# -------------------- 필터 UI 설정 --------------------
st.sidebar.title("🎛️ 추천 조건 설정")

all_genres = sorted(set(g for genres in df["genre_list"] for g in genres))
all_types = sorted(df["type"].dropna().unique())

selected_genres = st.sidebar.multiselect("🎭 장르", all_genres, default=["Action", "Comedy"])
selected_types = st.sidebar.multiselect("📺 형식", all_types)
rating_min, rating_max = st.sidebar.slider("⭐ 평점 범위", 0.0, 10.0, (6.0, 10.0), step=0.1)
members_min, members_max = st.sidebar.slider("👥 인기도 범위 (members)", 0, 1500000, (50000, 1000000), step=10000)
search_keyword = st.sidebar.text_input("🔍 제목 키워드 포함", "")

# -------------------- 필터링 함수 --------------------
def filter_anime(df, genres, types, r_min, r_max, m_min, m_max, keyword):
    filtered = df[
        (df["rating"].between(r_min, r_max)) &
        (df["members"].between(m_min, m_max))
    ]
    if types:
        filtered = filtered[filtered["type"].isin(types)]
    if keyword:
        filtered = filtered[filtered["name"].str.contains(keyword, case=False, na=False)]
    if genres:
        filtered = filtered[filtered["genre_list"].apply(lambda g_list: all(g in g_list for g in genres))]
    return filtered

filtered_df = filter_anime(df, selected_genres, selected_types,
                           rating_min, rating_max, members_min, members_max, search_keyword)

# -------------------- 이미지 출력 함수 (Jikan API + 기본 이미지 대체) --------------------
EXCLUDED_IMAGE_GENRES = {"Hentai", "Ecchi", "Horror", "Yaoi"}
DEFAULT_IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"

@st.cache_data(show_spinner=False)
def get_anime_image_url(title):
    try:
        response = requests.get("https://api.jikan.moe/v4/anime", params={"q": title, "limit": 1})
        if response.status_code == 200:
            data = response.json()
            if data["data"]:
                return data["data"][0]["images"]["jpg"]["image_url"]
    except:
        return None
    return None

# -------------------- 메인 출력 --------------------
st.title("🎌 애니메이션 추천기")
st.markdown("선택한 조건에 맞는 애니메이션을 추천해드립니다!")

if filtered_df.empty:
    st.warning("조건에 맞는 애니메이션이 없어요 😥\n필터를 조금 완화해 보세요.")
else:
    top_recommendations = filtered_df.sort_values(by="rating", ascending=False).head(10)

    st.subheader("📋 추천 애니메이션")
    for _, row in top_recommendations.iterrows():
        anime_name = row['name']
        genres = ", ".join(row['genre_list'])
        st.markdown(f"**🎬 {anime_name}**")
        st.markdown(f"⭐ 평점: {row['rating']} | 👥 Members: {row['members']} | 📺 Type: {row['type']}  \n🎭 장르: {genres}")

        genre_set = set(row["genre_list"])
        img_url = None
        if not genre_set.intersection(EXCLUDED_IMAGE_GENRES):
            img_url = get_anime_image_url(anime_name)

        st.image(img_url if img_url else DEFAULT_IMG_URL, width=200)
        st.markdown("---")

    # -------------------- Plotly 시각화 --------------------
    st.subheader("📊 평점 높은 순")
    fig_rating = px.bar(top_recommendations.sort_values(by="rating"),
                        y="name", x="rating", color="type",
                        orientation="h", title="평점 높은 애니메이션",
                        labels={"name": "애니메이션", "rating": "평점"})
    st.plotly_chart(fig_rating, use_container_width=True)

    st.subheader("📊 인기도 높은 순")
    fig_members = px.bar(top_recommendations.sort_values(by="members"),
                         y="name", x="members", color="type",
                         orientation="h", title="인기도 높은 애니메이션",
                         labels={"name": "애니메이션", "members": "Members"})
    st.plotly_chart(fig_members, use_container_width=True)
