"""
Movie & TV Recommender - Streamlit web interface
MSAI-631 AI for Human Computer Interaction, Tracy Ba-Taa-Banah

Replaces the original Tkinter desktop window
(https://github.com/shyam1998/Movie-Recommendation-System-GUI) with a browser-based interface
that adds filters and a "Why recommended?" explanation for each result.

Run with:  streamlit run app.py
"""
import streamlit as st

import recommender as R

st.set_page_config(page_title="Movie & TV Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading movies...")
def get_engine():
    return R.ContentRecommender(R.load_imdb())


engine = get_engine()
titles = engine.df["sortedTitle"].tolist()

# ----------------------------------------------------------------- sidebar: filters
with st.sidebar:
    st.header("⚙️ Filters")
    same_kind = st.checkbox("Only recommend the same type (movie vs. TV)", value=True)
    years = st.slider("Release years", 1960, 2020, (1960, 2020))
    min_rating = st.slider("Minimum IMDb rating", 0.0, 9.0, 0.0, 0.5)
    n = st.slider("Number of recommendations", 5, 20, 10)
    st.divider()
    st.caption("How it works: each title is described by its genres, rating, popularity, and era. "
               "The app recommends the titles most similar to your pick (cosine similarity).")

# ----------------------------------------------------------------- main page
st.title("🎬 Movie & TV Recommender")
st.write("Pick a movie or TV show you like, and I'll recommend what to watch next and explain why.")

choice = st.selectbox("Start typing a movie or TV show:", titles, index=None,
                      placeholder="e.g. Toy Story (1995)")

if choice:
    idx = titles.index(choice)
    results = engine.recommend(idx, n=n, same_kind=same_kind, year_range=years,
                               min_rating=min_rating)
    if results.empty:
        st.warning("No titles match these filters. Try widening the years or lowering the rating.")
    else:
        st.subheader(f"Because you liked {choice}:")
        for i, r in results.iterrows():
            with st.container(border=True):
                st.markdown(f"**{i + 1}. [{r.sortedTitle}]({r.url})** &nbsp; "
                            f"<span style='color:gray'>{r.kind} · {r.genres} · ⭐ {r.averageRating}"
                            f" · {r.similarity:.1%} match</span>", unsafe_allow_html=True)
                st.caption("Why recommended: " + r.why)

st.divider()
st.caption("Based on Movie-Recommendation-System-GUI by shyam1998 (MIT License). "
           "Data: IMDb sample from that project. Built for MSAI-631 by Tracy Ba-Taa-Banah.")
