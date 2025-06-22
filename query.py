import argparse
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def generate_with_llama(prompt: str) -> str:
    """Call LLaMA 3.1 8B on Hugging Face (via SambaNova)"""
    client = InferenceClient(
        provider="sambanova",
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )
    return completion.choices[0].message["content"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load local embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},  # or "cpu"
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load vector store
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    # Format prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    response_text = generate_with_llama(prompt)

    # Show results
    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"\nResponse: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
