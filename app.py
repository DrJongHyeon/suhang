import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["rating", "genre", "type"])
    df = df[df["episodes"].apply(lambda x: x.isdigit())]  # 숫자 에피소드만
    df["episodes"] = df["episodes"].astype(int)
    return df

df = load_data()

st.title("📊 애니메이션 인기 분석 대시보드")
st.markdown("분석 대상: [MyAnimeList 데이터셋](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)")

# 1. 평점 vs 인기도
fig1 = px.scatter(
    df, x="rating", y="members",
    hover_data=["name", "type"],
    title="⭐ 평점 vs 인기도",
    labels={"rating": "평점", "members": "인기도 (Members)"}
)
st.plotly_chart(fig1, use_container_width=True)

# 2. 에피소드 수 vs 인기도
fig2 = px.scatter(
    df, x="episodes", y="members",
    hover_data=["name", "type"],
    title="🎬 에피소드 수 vs 인기도",
    labels={"episodes": "에피소드 수", "members": "인기도 (Members)"}
)
st.plotly_chart(fig2, use_container_width=True)

# 3. 타입별 평균 인기도
type_avg = df.groupby("type")["members"].mean().reset_index().sort_values("members", ascending=False)
fig3 = px.bar(
    type_avg, x="type", y="members",
    title="📺 애니 타입별 평균 인기도",
    labels={"type": "애니 유형", "members": "평균 인기도"}
)
st.plotly_chart(fig3, use_container_width=True)

# 4. 장르별 평균 인기도 (장르 분할)
from collections import defaultdict
genre_members = defaultdict(list)

for _, row in df.iterrows():
    genres = row["genre"].split(", ")
    for g in genres:
        genre_members[g].append(row["members"])

genre_df = pd.DataFrame({
    "genre": list(genre_members.keys()),
    "avg_members": [sum(vals)/len(vals) for vals in genre_members.values()]
}).sort_values("avg_members", ascending=False)

fig4 = px.bar(
    genre_df, x="genre", y="avg_members",
    title="🎭 장르별 평균 인기도",
    labels={"genre": "장르", "avg_members": "평균 인기도"}
)
st.plotly_chart(fig4, use_container_width=True)

# 결론 요약
st.markdown("""
### 🔍 요약
- 평점이 높을수록 인기도도 증가하는 경향이 있음
- TV 시리즈가 가장 높은 평균 인기도를 보임
- 장르별로는 `Action`, `Shounen`, `Drama` 등이 인기 높은 경향
""")
