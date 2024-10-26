from tqdm import tqdm
import pandas as pd
import requests
import re

def pure_string(s):
    allowed = r'[^a-zA-Z0-9()"\'`\-_%!?\$@&:.,]'
    regex = re.compile(allowed)
    content = regex.sub(' ', s.lower())
    return " ".join(content.split())

def is_valid_single_char(word):
    return (word.isdigit() or
            word.lower() in ['i', 'u', 'w', 'x', 'n', 'r'])

def should_process_review(text):
    """
    Check if review should be processed based on rules:
    1. At least 3 words
    2. No single char words except numbers and [i,u,w,x,n,r]
    """
    new_text = pure_string(text)
    words = new_text.split()
    new_words = []
    for word in words:
        if len(word) == 1 and not is_valid_single_char(word):
            continue
        new_words.append(word)

    if len(new_words) < 3:
        return False

    return True

BASE_URL = "http://localhost:8092"
ENDPOINT = "review"

# fname = "sample.csv"
fname = "../SPOTIFY_REVIEWS.csv"
df = pd.read_csv(fname)[:400000]
df = df.dropna(subset=["review_text"])
df['author_app_version'] = df['author_app_version'].fillna('0')
# print(df)
# print(df.keys())

for _id, row in tqdm(df.iterrows(), total=len(df)):
    if _id < 224304:
        continue
    review_text = row["review_text"]

    # Check if review should be processed
    if not should_process_review(review_text):
        # print(f"SKIPPED REVIEW TEXT: {review_text}")
        continue

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
    except Exception as e:
        print(f"FAILED DURING DATA INSERTION: {e}")
        print(row)
        break


