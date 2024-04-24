import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingModel:
    def __init__(self):
        try:
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def embed(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        return self.model(texts).numpy()

    def compute_similarity(self, embedding1, embedding2):
        # Reshape to 1D array
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]
