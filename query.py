import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient, HfApi
import os

# 🔐 Fallback token (will be used only if env variable not found)
FALLBACK_HF_TOKEN = "hf_your_actual_token_here"

# Chroma DB path
CHROMA_PATH = "chroma"

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_hf_token():
    return os.environ.get("HF_TOKEN", FALLBACK_HF_TOKEN)

def verify_hf_login(token):
    try:
        who = HfApi(token=token).whoami()
        print(f"✅ Logged in to Hugging Face as: {who['name']}")
    except Exception as e:
        print("❌ Hugging Face login failed. Check your HF_TOKEN.")
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    hf_token = get_hf_token()
    verify_hf_login(hf_token)

    # Local embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("⚠️ No relevant documents found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    # Hugging Face inference
    client = InferenceClient(provider="nscale", api_key=hf_token)

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.choices[0].message["content"]
    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"📘 Response:\n{response_text}\n\n📎 Sources:\n{sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
