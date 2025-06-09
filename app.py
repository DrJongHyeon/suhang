import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 데이터 불러오기 --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["genre", "rating", "type", "name"])
    df = df[df["episodes"].apply(lambda x: x.isdigit())]
    df["episodes"] = df["episodes"].astype(int)
    df["genre"] = df["genre"].str.strip()
    return df

df = load_data()

# -------------------- 장르 전처리 --------------------
all_genres = sorted(set(g for gs in df["genre"] for g in gs.split(", ")))

# -------------------- 사이드바 --------------------
st.sidebar.title("🎛️ 추천 조건 설정")
selected_genres = st.sidebar.multiselect("🎭 장르", all_genres, default=["Action", "Comedy"])
selected_type = st.sidebar.selectbox("📺 형식", sorted(df["type"].unique()))
min_rating = st.sidebar.slider("⭐ 최소 평점", 0.0, 10.0, 7.0, 0.1)
min_members = st.sidebar.slider("👥 최소 인기도 (members)", 0, 1000000, 50000, step=10000)
search_keyword = st.sidebar.text_input("🔍 제목 키워드 포함", "")

st.title("🎌 애니메이션 추천기")
st.markdown("조건에 맞는 애니메이션을 추천하고, 유사한 작품도 찾아드릴게요!")

# -------------------- 필터링 함수 --------------------
def filter_anime(df, genres, anime_type, min_rating, min_members, keyword):
    filtered = df[
        (df["type"] == anime_type) &
        (df["rating"] >= min_rating) &
        (df["members"] >= min_members)
    ]
    if keyword:
        filtered = filtered[filtered["name"].str.contains(keyword, case=False, na=False)]

    def has_genres(genre_str):
        genre_set = set(genre_str.split(", "))
        return all(g in genre_set for g in genres)

    filtered = filtered[filtered["genre"].apply(has_genres)]
    return filtered

# -------------------- 필터 적용 --------------------
filtered_df = filter_anime(df, selected_genres, selected_type, min_rating, min_members, search_keyword)
top_recommendations = filtered_df.sort_values(by="rating", ascending=False).head(10)

# -------------------- 결과 출력 --------------------
st.subheader("📋 추천 애니메이션")
if top_recommendations.empty:
    st.warning("조건에 맞는 애니메이션이 없어요 😥")
else:
    for _, row in top_recommendations.iterrows():
        st.markdown(f"**🎬 {row['name']}**  \n"
                    f"⭐ 평점: {row['rating']} | 👥 Members: {row['members']} | 📺 Type: {row['type']}  \n"
                    f"🎭 장르: {row['genre']}  \n"
                    "---")

# -------------------- Plotly 시각화 --------------------
if not top_recommendations.empty:
    fig = px.scatter(
        top_recommendations,
        x="rating", y="members",
        hover_data=["name"],
        color="type",
        title="📊 추천된 애니의 평점 vs 인기도"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- 사용자 선호 기반 유사도 추천 --------------------
st.subheader("🤝 유사한 애니메이션 추천 (선호 기반)")

if not top_recommendations.empty:
    # 사용자 입력을 첫 번째 추천 작품으로 간주
    target = top_recommendations.iloc[0]

    # 벡터화: 장르 + 타입 + 평점 범주화
    df["features"] = df["genre"] + " " + df["type"] + " rating_" + df["rating"].round().astype(str)
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(df["features"])

    # 코사인 유사도 계산
    index = df[df["name"] == target["name"]].index[0]
    similarity = cosine_similarity(feature_matrix[index], feature_matrix).flatten()
    df["similarity"] = similarity

    similar_df = df[df["name"] != target["name"]].sort_values("similarity", ascending=False).head(5)

    for _, row in similar_df.iterrows():
        st.markdown(f"**🔁 {row['name']}**  \n"
                    f"⭐ 평점: {row['rating']} | 👥 Members: {row['members']}  \n"
                    f"🎭 장르: {row['genre']} | 📺 Type: {row['type']}  \n"
                    "---")
