import streamlit as st
import pandas as pd
import plotly.express as px

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
rating_min, rating_max = st.sidebar.slider("⭐ 평점 범위", 0.0, 10.0, (6.0, 10.0), step=0.1)
members_min, members_max = st.sidebar.slider("👥 인기도 범위 (members)", 0, 1500000, (50000, 1000000), step=10000)
search_keyword = st.sidebar.text_input("🔍 제목 키워드 포함", "")

st.title("🎌 애니메이션 추천기")
st.markdown("조건에 맞는 애니메이션을 추천해드립니다!")

# -------------------- 필터링 함수 --------------------
def filter_anime(df, genres, types, r_min, r_max, m_min, m_max, keyword):
    filtered = df[
        (df["type"].isin(types)) &
        (df["rating"] >= r_min) & (df["rating"] <= r_max) &
        (df["members"] >= m_min) & (df["members"] <= m_max)
    ]
    if keyword:
        filtered = filtered[filtered["name"].str.contains(keyword, case=False, na=False)]

    def has_genres(genre_str):
        genre_set = set(genre_str.split(", "))
        return all(g in genre_set for g in genres)

    filtered = filtered[filtered["genre"].apply(has_genres)]
    return filtered

# -------------------- 필터 적용 --------------------
filtered_df = filter_anime(df, selected_genres, selected_types,
                           rating_min, rating_max, members_min, members_max, search_keyword)

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

    # -------------------- Plotly 가로 막대 시각화 --------------------
    st.subheader("📊 평점 높은 순")
    rating_bar = top_recommendations.sort_values(by="rating", ascending=True)
    fig_rating = px.bar(rating_bar, y="name", x="rating", color="type",
                        orientation="h", title="평점 상위 애니메이션",
                        labels={"name": "애니메이션", "rating": "평점"})
    st.plotly_chart(fig_rating, use_container_width=True)

    st.subheader("📊 인기도 높은 순")
    members_bar = top_recommendations.sort_values(by="members", ascending=True)
    fig_members = px.bar(members_bar, y="name", x="members", color="type",
                         orientation="h", title="인기도 상위 애니메이션",
                         labels={"name": "애니메이션", "members": "Members"})
    st.plotly_chart(fig_members, use_container_width=True)
