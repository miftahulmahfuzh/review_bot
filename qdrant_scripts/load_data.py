import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "http://localhost:8092"
ENDPOINT = "review"

# fname = "sample.csv"
fname = "../SPOTIFY_REVIEWS.csv"
df = pd.read_csv(fname)[:10000] # [:5]
df = df.dropna(subset=["review_text"])
df['author_app_version'] = df['author_app_version'].fillna('1.1.0.112')
# print(df)
# print(df.keys())

for i, row in tqdm(df.iterrows(), total=len(df)):
    review_text = row["review_text"]
    _id = int(row["Unnamed: 0"])
    payload = {
        "_id": _id,
        "review_id": row["review_id"],
        "pseudo_author_id": row["pseudo_author_id"],
        "author_name": row["author_name"],
        "review_text": review_text,
        "review_rating": int(row["review_rating"]),
        "review_likes": int(row["review_likes"]),
        "author_app_version": row["author_app_version"],
        "review_timestamp": row["review_timestamp"]
    }
    try:
        response = requests.put(
            f"{BASE_URL}/{ENDPOINT}/{_id}",
            json=payload,
            headers={"Content-Type": "application/json", "x-api-key": "ebce2698dadf0593c979a2798c84e49a0"}
        )
    except:
        print(row)


