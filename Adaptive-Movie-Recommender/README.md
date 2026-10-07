# Adaptive Movie & TV Recommender

**MSAI-631 AI for Human Computer Interaction** · Tracy Ba-Taa-Banah · University of the Cumberlands

An **AI-based adaptive user interface**. The app learns each user's taste from 👍/👎 feedback while
it is running and changes what it recommends and how the interface looks in response.

## Starting point and credit

- **Original system:** [Movie-Recommendation-System-GUI](https://github.com/shyam1998/Movie-Recommendation-System-GUI)
  by Shyam (MIT License), a content-based IMDb recommender with a Tkinter desktop window.
- **My previous version:** I extended that project into a Streamlit web app with corrected feature
  scaling, filters, and explanations
  ([folder](https://github.com/Tracybat16/ai-hci-coursework/tree/main/Movie-Recommendation-System-GUI)).
- **This project:** adds a runtime adaptation layer (`adaptive.py`) and an adaptive interface (`app.py`).

## How the app adapts

| User action | What the system learns | How the interface adapts |
|---|---|---|
| 👍 or 👎 on a recommendation | Genre preference weights go up (+0.25) or down (−0.25), clipped to [−1, 1] | Results are **re-ranked**: final score = similarity + 0.15 × genre preference. Titles show *moved up/down* badges and "Matches genres you liked." |
| Any feedback | Running taste profile | A **"Your taste so far"** panel appears with top liked and disliked genres. |
| 2+ likes | Years of liked titles | The **release-year filter adjusts itself** to the user's preferred range (±5 years). |
| 3+ likes | Average feature vector of liked titles | A **"Picked for you"** section appears automatically (proactive suggestions). |
| — | — | Every change is logged in **"How the app has adapted."** |

**User control:** adaptation can be switched off, the year filter can be kept manual, and
**Reset** clears everything the app has learned. Learning lasts only for the current browser session;
nothing is stored.

## How to run

```bash
conda activate recommender      # or: pip install -r requirements.txt
streamlit run app.py
```

## Files

```
app.py          Streamlit interface with adaptive panels (new)
adaptive.py     User model: online preference learning and adaptation rules (new)
recommender.py  Content-based engine from my previous project
dataset/imdb_sampled.csv  14,051 IMDb titles (from the original project)
```

## Use of AI tools

Claude (Anthropic, Opus 5.5) generated the code in `adaptive.py` and `app.py` and helped write this
README based on my design requirements. I ran, tested, and verified the application.
