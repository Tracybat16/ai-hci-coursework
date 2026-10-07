"""
Adaptive Movie & TV Recommender - Streamlit web interface
MSAI-631 AI for Human Computer Interaction, Tracy Ba-Taa-Banah

Builds on my Movie & TV Recommender (which extends shyam1998/Movie-Recommendation-System-GUI).
New in this version: the interface adapts to the user at runtime.
  - 👍 / 👎 feedback trains a genre-preference model that re-ranks recommendations
  - a "Your taste so far" panel appears once the system has learned something
  - the release-year filter can adjust itself to the years the user tends to like
  - "Picked for you" suggestions appear automatically after 3 likes
  - every adaptation is explained, and the user can switch adaptation off or reset it

Run with:  streamlit run app.py
"""
import streamlit as st

import recommender as R
from adaptive import UserModel, MIN_LIKES_FOR_PICKS

st.set_page_config(page_title="Adaptive Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading movies...")
def get_engine():
    return R.ContentRecommender(R.load_imdb())


engine = get_engine()
df = engine.df
titles = df["sortedTitle"].tolist()

if "user" not in st.session_state:
    st.session_state.user = UserModel(engine)
if "years" not in st.session_state:
    st.session_state.years = (1960, 2020)
user = st.session_state.user

# ------------------------------------------------ adapt the year filter before it is drawn
adapt_on = st.session_state.get("adapt_on", True)
auto_years = st.session_state.get("auto_years", True)
suggested = user.suggested_years()
if adapt_on and auto_years and suggested and st.session_state.get("last_suggested") != suggested:
    st.session_state.years = suggested
    st.session_state.last_suggested = suggested
    user.log.append(f"Release-year filter adjusted to {suggested[0]}–{suggested[1]} "
                    "to match the titles you liked.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Filters")
    same_kind = st.checkbox("Only recommend the same type (movie vs. TV)", value=True)
    years = st.slider("Release years", 1960, 2020, key="years")
    min_rating = st.slider("Minimum IMDb rating", 0.0, 9.0, 0.0, 0.5)
    n = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.header("🧠 Adaptation")
    st.toggle("Learn from my feedback", value=True, key="adapt_on")
    st.checkbox("Let the year filter adjust itself", value=True, key="auto_years")
    st.caption(f"👍 {len(user.liked)} liked · 👎 {len(user.disliked)} disliked")
    if st.button("Reset what the app has learned"):
        user.reset()
        st.session_state.years = (1960, 2020)
        st.session_state.last_suggested = None
        st.rerun()

# ------------------------------------------------------------------ main page
st.title("🎬 Adaptive Movie & TV Recommender")
st.write("Pick a title, then rate the recommendations with 👍 or 👎. "
         "The app learns your taste as you go and adapts what it shows you.")

# Adaptive panel 1: what the system has learned (appears only after feedback)
if adapt_on and user.has_learned():
    with st.container(border=True):
        st.subheader("🧠 Your taste so far")
        likes, dislikes = user.top_genres(3, True), user.top_genres(3, False)
        c1, c2 = st.columns(2)
        c1.markdown("**More of:** " + (", ".join(f"{g} (+{w:.2f})" for g, w in likes) or "—"))
        c2.markdown("**Less of:** " + (", ".join(f"{g} ({w:.2f})" for g, w in dislikes) or "—"))
        if user.log:
            with st.expander("How the app has adapted"):
                for line in reversed(user.log[-8:]):
                    st.write("• " + line)

# Adaptive panel 2: proactive suggestions (appear after enough likes)
picks = user.proactive_picks() if adapt_on else None
if picks is not None:
    st.subheader("✨ Picked for you")
    st.caption(f"Based on the {len(user.liked)} titles you liked, with no search needed.")
    cols = st.columns(3)
    for k, r in picks.iterrows():
        cols[k % 3].markdown(f"**[{r.sortedTitle}]({r.url})**  \n"
                             f"<span style='color:gray'>{r.genres} · ⭐ {r.averageRating}</span>",
                             unsafe_allow_html=True)
    st.divider()
elif adapt_on and user.liked:
    left = MIN_LIKES_FOR_PICKS - len(user.liked)
    st.info(f"Like {left} more title{'s' if left > 1 else ''} and I'll start suggesting "
            "picks for you automatically.")

choice = st.selectbox("Start typing a movie or TV show:", titles, index=None,
                      placeholder="e.g. Toy Story (1995)")

if choice:
    idx = titles.index(choice)
    results = engine.recommend(idx, n=n, same_kind=same_kind, year_range=years,
                               min_rating=min_rating)
    if adapt_on:
        results = user.personalize(results)
    if results.empty:
        st.warning("No titles match these filters. Try widening the years or lowering the rating.")
    else:
        label = "personalized for you" if adapt_on and user.has_learned() else "not personalized yet"
        st.subheader(f"Because you liked {choice} ({label}):")
        for i, r in results.iterrows():
            ridx = engine.find(r.sortedTitle)
            with st.container(border=True):
                c1, c2, c3 = st.columns([8, 1, 1])
                with c1:
                    badge = ""
                    if adapt_on and r.get("moved", 0) > 0:
                        badge = f" &nbsp;⬆️ <span style='color:green'>moved up {int(r.moved)}</span>"
                    elif adapt_on and r.get("moved", 0) < 0:
                        badge = f" &nbsp;⬇️ <span style='color:#b45309'>moved down {int(-r.moved)}</span>"
                    st.markdown(f"**{i + 1}. [{r.sortedTitle}]({r.url})** &nbsp; "
                                f"<span style='color:gray'>{r.kind} · {r.genres} · ⭐ {r.averageRating}"
                                f" · {r.similarity:.1%} match</span>{badge}",
                                unsafe_allow_html=True)
                    why = "Why recommended: " + r.why
                    if adapt_on and r.get("personal", 0) > 0:
                        why += " · Matches genres you liked"
                    elif adapt_on and r.get("personal", 0) < 0:
                        why += " · Includes genres you disliked"
                    st.caption(why)
                rated = ridx in user.liked or ridx in user.disliked
                if c2.button("👍", key=f"up_{r.tconst}", disabled=rated or not adapt_on):
                    user.feedback(ridx, True)
                    st.rerun()
                if c3.button("👎", key=f"down_{r.tconst}", disabled=rated or not adapt_on):
                    user.feedback(ridx, False)
                    st.rerun()

st.divider()
st.caption("Based on Movie-Recommendation-System-GUI by shyam1998 (MIT License). "
           "Adaptive features built for MSAI-631 by Tracy Ba-Taa-Banah.")
