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

# -------------------- 장르 및 형식 목록 --------------------
all_genres = sorted(set(g for gs in df["genre"] for g in gs.split(", ")))
all_types = sorted(df["type"].unique())

# -------------------- 사이드바 필터 --------------------
st.sidebar.title("🎛️ 추천 조건 설정")
selected_genres = st.sidebar.multiselect("🎭 장르", all_genres, default=["Action", "Comedy"])
selected_types = st.sidebar.multiselect("📺 형식", all_types, default=["TV"])
min_rating = st.sidebar.slider("⭐ 최소 평점", 0.0, 10.0, 7.0, step=0.1)
min_members = st.sidebar.slider("👥 최소 인기도 (members)", 0, 1000000, 50000, step=10000)
search_keyword = st.sidebar.text_input("🔍 제목 키워드 포함", "")

st.title("🎌 애니메이션 추천기")
st.markdown("조건에 맞는 애니메이션을 추천하고, 유사한 작품도 찾아드릴게요!")

# -------------------- 필터링 함수 --------------------
def filter_anime(df, genres, types, min_rating, min_members, keyword):
    filtered = df[
        (df["type"].isin(types)) &
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

# -------------------- 추천 목록 만들기 --------------------
filtered_df = filter_anime(df, selected_genres, selected_types, min_rating, min_members, search_keyword)

if not filtered_df.empty:
    top_recommendations = filtered_df.sort_values(by="rating", ascending=False).head(10)
else:
    top_recommendations = pd.DataFrame()

# -------------------- 추천 출력 --------------------
st.subheader("📋 추천 애니메이션")
if top_recommendations.empty:
    st.warning("조건에 맞는 애니메이션이 없어요 😥\n필터를 조금 완화해 보세요.")
else:
    for _, row in top_recommendations.iterrows():
        st.markdown(f"**🎬 {row['name']}**  \n"
                    f"⭐ 평점: {row['rating']} | 👥 Members: {row['members']} | 📺 Type: {row['type']}  \n"
                    f"🎭 장르: {row['genre']}  \n"
                    "---")

    # -------------------- Plotly 시각화 --------------------
    st.subheader("📊 추천 애니메이션의 평점 순위")
    rating_bar = top_recommendations.sort_values(by="rating", ascending=False)
    fig_rating = px.bar(rating_bar, x="name", y="rating", color="type", title="평점 높은 순", labels={"name": "애니메이션"})
    fig_rating.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_rating, use_container_width=True)

    st.subheader("📊 추천 애니메이션의 인기도 순위")
    members_bar = top_recommendations.sort_values(by="members", ascending=False)
    fig_members = px.bar(members_bar, x="name", y="members", color="type", title="인기도 높은 순", labels={"name": "애니메이션"})
    fig_members.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_members, use_container_width=True)

    # -------------------- 유사도 기반 추천 --------------------
    st.subheader("🤝 유사한 애니메이션 추천")

    # 기준 작품: top 1
    target = top_recommendations.iloc[0]

    # 벡터화
    df["features"] = df["genre"] + " " + df["type"] + " rating_" + df["rating"].round().astype(str)
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(df["features"])

    # 유사도 계산
    idx = df[df["name"] == target["name"]].index[0]
    similarity = cosine_similarity(feature_matrix[idx], feature_matrix).flatten()
    df["similarity"] = similarity

    similar_df = df[df["name"] != target["name"]].sort_values("similarity", ascending=False).head(5)

    for _, row in similar_df.iterrows():
        st.markdown(f"**🔁 {row['name']}**  \n"
                    f"⭐ 평점: {row['rating']} | 👥 Members: {row['members']}  \n"
                    f"🎭 장르: {row['genre']} | 📺 Type: {row['type']}  \n"
                    "---")
