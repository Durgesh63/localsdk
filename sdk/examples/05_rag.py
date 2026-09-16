"""Example 5 -- retrieval-augmented generation (RAG) with embeddings.

The full loop: embed your documents once, embed the question, find the closest
documents by cosine similarity, and paste those into a chat prompt so the model
answers from your data instead of from memory.

Deliberately dependency-free -- the similarity search is plain Python. A few
thousand chunks do not need a vector database. Past that, look at Chroma or
pgvector; the embedding half of this file stays the same either way.

Requires the embedding model on the server's Ollama:
    ollama pull nomic-embed-text

Run:
    python examples/05_rag.py
"""

from math import sqrt

from localsdk import Client, LocalSDKError, NotFoundError

# Stand-in for your real corpus. In practice you would chunk documents into
# passages of a few hundred words each.
DOCUMENTS = [
    "The Pune office is open from 9am to 6pm, Monday through Friday.",
    "Expense reports must be filed within 30 days of the expense date.",
    "The VPN requires hardware-key two-factor authentication to connect.",
    "Parking passes are issued by facilities on the second floor.",
    "Laptops are replaced on a three-year cycle, or sooner if hardware fails.",
]

QUESTION = "How long do I have to submit an expense?"

TOP_K = 2


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1].

    Input:  cosine([1.0, 0.0], [1.0, 0.0])
    Output: 1.0
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def main() -> None:
    client = Client()

    # 1. Index. Do this ONCE and persist the vectors -- re-embedding an
    #    unchanged corpus on every run wastes time and metered tunnel requests.
    #    Send the whole list in one call rather than looping per document.
    print(f"embedding {len(DOCUMENTS)} documents...")
    doc_vectors = client.embed(DOCUMENTS)
    print(f"  {len(doc_vectors)} vectors of dimension {len(doc_vectors[0])}\n")

    # 2. Embed the question with the same model, so the vectors are comparable.
    question_vector = client.embed(QUESTION)[0]

    # 3. Retrieve: rank documents by similarity, keep the best few.
    ranked = sorted(
        zip(DOCUMENTS, doc_vectors),
        key=lambda pair: cosine(question_vector, pair[1]),
        reverse=True,
    )
    context = [doc for doc, _ in ranked[:TOP_K]]

    print(f"question: {QUESTION}")
    print("retrieved:")
    for doc in context:
        print(f"  - {doc}")
    print()

    # 4. Generate, grounded in what was retrieved. Telling the model to say it
    #    does not know is what stops it inventing an answer when retrieval misses.
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you do not know.\n\n"
        "Context:\n" + "\n".join(f"- {doc}" for doc in context) + f"\n\nQuestion: {QUESTION}"
    )

    answer = client.chat([{"role": "user", "content": prompt}])
    print("answer:", answer.text)

    client.close()


if __name__ == "__main__":
    try:
        main()
    except NotFoundError:
        print(
            "The embedding model is not installed on the server.\n"
            "On the machine running Ollama:  ollama pull nomic-embed-text"
        )
    except LocalSDKError as exc:
        print(f"{type(exc).__name__}: {exc}")
