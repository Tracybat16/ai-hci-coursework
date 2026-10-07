"""
Adaptive layer for the Movie & TV Recommender
MSAI-631 AI for Human Computer Interaction, Tracy Ba-Taa-Banah

This module learns from the user's interactions while the app is running and uses what it
learns to change both the recommendations and the interface.

How it learns (online preference learning):
  - Every genre starts with a preference weight of 0.
  - A thumbs-up adds LEARNING_RATE to each genre of the liked title; a thumbs-down subtracts it.
  - Weights are clipped to [-1, 1] so no single click dominates.
  - Recommendations are re-ranked with:  final score = similarity + PERSONAL_WEIGHT * genre preference
  - Liked titles also build a "taste vector" (the average of their feature vectors), which drives
    proactive "Picked for you" suggestions.
  - Release years of liked titles are tracked so the year filter can adapt.
"""
import numpy as np
from sklearn.preprocessing import normalize

LEARNING_RATE = 0.25
PERSONAL_WEIGHT = 0.15
MIN_LIKES_FOR_PICKS = 3
MIN_LIKES_FOR_YEARS = 2


class UserModel:
    def __init__(self, engine):
        self.engine = engine
        self.genres = list(engine.genre_names)
        self.weights = {g: 0.0 for g in self.genres}
        self.liked, self.disliked = [], []
        self.log = []  # plain-language record of every adaptation

    # ---------------------------------------------------------- learning
    def _genres_of(self, idx):
        row = self.engine.genre_matrix[idx]
        return [g for g, v in zip(self.genres, row) if v > 0]

    def feedback(self, idx, liked):
        title = self.engine.df["sortedTitle"].iat[idx]
        if idx in self.liked or idx in self.disliked:
            return
        (self.liked if liked else self.disliked).append(idx)
        sign = 1 if liked else -1
        for g in self._genres_of(idx):
            self.weights[g] = float(np.clip(self.weights[g] + sign * LEARNING_RATE, -1, 1))
        verb = "liked" if liked else "disliked"
        self.log.append(f"You {verb} {title}; adjusted weights for "
                        + ", ".join(self._genres_of(idx)) + ".")

    def reset(self):
        self.__init__(self.engine)

    # ---------------------------------------------------------- using what was learned
    def genre_preference(self):
        """Preference score for every title: average learned weight of its genres."""
        w = np.array([self.weights[g] for g in self.genres])
        counts = self.engine.genre_matrix.sum(axis=1)
        counts[counts == 0] = 1
        return (self.engine.genre_matrix @ w) / counts

    def personalize(self, results):
        """Re-rank a results table using the learned genre preferences."""
        if not self.has_learned():
            results["personal"] = 0.0
            results["final"] = results["similarity"]
            return results
        pref = self.genre_preference()
        idxs = [self.engine.find(t) for t in results["sortedTitle"]]
        results = results.copy()
        results["personal"] = [pref[i] for i in idxs]
        results["final"] = results["similarity"] + PERSONAL_WEIGHT * results["personal"]
        results["moved"] = 0
        old_rank = {t: r for r, t in enumerate(results["sortedTitle"])}
        results = results.sort_values("final", ascending=False).reset_index(drop=True)
        results["moved"] = [old_rank[t] - r for r, t in enumerate(results["sortedTitle"])]
        return results

    def has_learned(self):
        return any(abs(v) > 0 for v in self.weights.values())

    def top_genres(self, n=3, positive=True):
        items = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=positive)
        return [(g, w) for g, w in items[:n] if (w > 0 if positive else w < 0)]

    def suggested_years(self):
        """Year range the user seems to prefer, or None if not enough evidence yet."""
        if len(self.liked) < MIN_LIKES_FOR_YEARS:
            return None
        years = self.engine.df["startYear"].iloc[self.liked].to_numpy()
        lo = int(max(1960, years.min() - 5))
        hi = int(min(2020, years.max() + 5))
        return (lo, hi)

    def proactive_picks(self, n=6):
        """Suggestions built from everything the user liked, shown without a request."""
        if len(self.liked) < MIN_LIKES_FOR_PICKS:
            return None
        taste = normalize(self.engine.matrix[self.liked].mean(axis=0, keepdims=True))[0]
        scores = self.engine.matrix @ taste + PERSONAL_WEIGHT * self.genre_preference()
        scores[self.liked + self.disliked] = -np.inf
        order = np.argsort(-scores)[:n]
        cols = ["tconst", "sortedTitle", "kind", "genres", "averageRating"]
        out = self.engine.df.iloc[order][cols].copy()
        out["url"] = "https://www.imdb.com/title/" + out["tconst"] + "/"
        return out.reset_index(drop=True)
