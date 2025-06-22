from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

def main():
    # Use local MiniLM model
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}  # or "cpu"
    )

    # Define two words to compare
    word1 = "apple"
    word2 = "iphone"

    # Get embedding vectors
    vec1 = embedding_function.embed_query(word1)
    vec2 = embedding_function.embed_query(word2)

    print(f"Vector for '{word1}' (length={len(vec1)}):\n{vec1[:10]}...")
    print(f"Vector for '{word2}' (length={len(vec2)}):\n{vec2[:10]}...")

    # Compare using cosine distance
    distance = cosine_distances([vec1], [vec2])[0][0]
    similarity = 1 - distance

    print(f"\nCosine Distance: {distance:.4f}")
    print(f"Cosine Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
