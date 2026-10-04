"""
Movie & TV Recommender - recommendation engine
MSAI-631 AI for Human Computer Interaction, Tracy Ba-Taa-Banah

Based on the content-based IMDb recommender by Shyam
(https://github.com/shyam1998/Movie-Recommendation-System-GUI, MIT License).

How it works (content-based filtering):
  Each title is turned into a list of numbers describing it: its genres, its IMDb rating,
  how many people voted, and its decade. Titles whose lists point in the same direction
  (high cosine similarity) are considered alike and are recommended.

Changes from the original:
  1. Fixed feature scaling: the original placed the raw 1-10 rating next to 0/1 genre flags,
     so rating dominated similarity. All features are now scaled to 0-1, and genres are weighted.
  2. Added a decade feature so recommendations respect era.
  3. Similarity is computed on demand for the chosen title instead of a full
     14,000 x 14,000 matrix (faster start-up and far less memory).
  4. Added filters (same type, release years, minimum rating).
  5. Added a plain-language explanation for every recommendation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMDB_PATH = os.path.join(BASE_DIR, "dataset", "imdb_sampled.csv")
TV_TYPES = {"tvSeries", "tvMovie", "tvMiniSeries", "video", "tvSpecial"}


def load_imdb(path=IMDB_PATH):
    df = pd.read_csv(path)
    df = df.drop_duplicates("tconst").reset_index(drop=True)
    df["genres"] = df["genres"].fillna("")
    df["decade"] = (df["startYear"] // 10 * 10).astype(int)
    df["kind"] = np.where(df["titleType"].isin(TV_TYPES), "TV", "Movie")
    return df


class ContentRecommender:
    def __init__(self, df, genre_weight=1.5):
        self.df = df
        # 1) genres -> one 0/1 column per genre (e.g., Action, Comedy, Sci-Fi)
        cv = CountVectorizer(token_pattern=r"[A-Za-z\-]+", lowercase=False, binary=True)
        self.genre_matrix = cv.fit_transform(df["genres"]).toarray().astype(float)
        self.genre_names = np.array(cv.get_feature_names_out())
        # 2) scale the number features to 0-1 so no single feature dominates
        scaler = MinMaxScaler()
        rating = scaler.fit_transform(df[["averageRating"]])
        votes = scaler.fit_transform(np.log1p(df[["numVotes"]]))  # log: votes range from 5 to 2 million
        decade = scaler.fit_transform(df[["decade"]])
        features = np.hstack([self.genre_matrix * genre_weight, rating, votes, decade])
        # 3) make every row length 1, so a dot product equals cosine similarity
        self.matrix = normalize(features)

    def find(self, title):
        t = title.strip().lower()
        hit = self.df.index[self.df["sortedTitle"].str.lower() == t]
        if len(hit) == 0:
            hit = self.df.index[self.df["primaryTitle"].str.lower() == t]
        return int(hit[0]) if len(hit) else None

    def recommend(self, idx, n=10, same_kind=True, year_range=None, min_rating=0.0):
        scores = self.matrix @ self.matrix[idx]  # cosine similarity to the chosen title

        mask = np.ones(len(self.df), dtype=bool)
        mask[idx] = False  # don't recommend the title itself
        if same_kind:
            mask &= (self.df["kind"] == self.df["kind"].iat[idx]).to_numpy()
        if year_range:
            mask &= self.df["startYear"].between(*year_range).to_numpy()
        mask &= (self.df["averageRating"] >= min_rating).to_numpy()

        order = np.argsort(-np.where(mask, scores, -np.inf))[:n]
        out = self.df.iloc[order][["tconst", "sortedTitle", "kind", "genres",
                                   "averageRating", "numVotes", "startYear"]].copy()
        out["similarity"] = scores[order]
        out["why"] = [self.explain(idx, j) for j in order]
        out["url"] = "https://www.imdb.com/title/" + out["tconst"] + "/"
        return out.reset_index(drop=True)

    def explain(self, src, rec):
        reasons = []
        shared = self.genre_names[(self.genre_matrix[src] > 0) & (self.genre_matrix[rec] > 0)]
        if len(shared):
            reasons.append("Shares genres: " + ", ".join(shared))
        reasons.append(f"Rated {self.df['averageRating'].iat[rec]:.1f}/10 by "
                       f"{self.df['numVotes'].iat[rec]:,} IMDb users")
        if self.df["decade"].iat[src] == self.df["decade"].iat[rec]:
            reasons.append(f"Same era ({self.df['decade'].iat[rec]}s)")
        return " · ".join(reasons)


if __name__ == "__main__":
    # text-mode demo:  python recommender.py "The Matrix (1999)"
    import sys
    engine = ContentRecommender(load_imdb())
    title = sys.argv[1] if len(sys.argv) > 1 else "Toy Story (1995)"
    i = engine.find(title)
    if i is None:
        print("Title not found.")
    else:
        print(f"Because you liked {engine.df['sortedTitle'].iat[i]}:")
        for _, r in engine.recommend(i).iterrows():
            print(f" - {r.sortedTitle}  [{r.why}]")
