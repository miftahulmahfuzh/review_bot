# insert instance to qdrant one by one

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct
import json
from tqdm import tqdm
import numpy as np

MODE = "hoax"
# MODE = "fact"

# Initialize Qdrant client
host = "localhost"
# host = "10.181.131.250"
client = QdrantClient(host, port=6333)

# Define collection name
collection_name = f"climate_{MODE}"

# Load vectors from JSON file
with open(f'samples/{MODE}.json', 'r') as f:
    data = json.load(f)

# Determine vector size from the first item
vector_size = len(data['items'][0]['vector'])
print(f"VECTOR SIZE: {vector_size}")

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)

# Prepare points for insertion
points = []
i = 0
print(f"Inserting points to collection: {collection_name}")
for item in tqdm(data['items']):
    vector = np.array(item['vector'], dtype=np.float32)
    client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        'title': item['title'],
                        'content': item['content'],
                        'url': item['url']
                    },
                )
            ],
        )
    i += 1
