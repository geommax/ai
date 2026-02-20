"""
06 - Retrieval
Vector store ထဲက relevant documents တွေကို retrieve လုပ်တယ်။
"""

from langchain_chroma import Chroma


def get_retriever(vectorstore: Chroma, top_k: int = 3):
    """
    Vector store ကနေ retriever object ကို တည်ဆောက်ပေးတယ်။

    Args:
        vectorstore: Chroma vector store instance
        top_k: retrieve လုပ်မယ့် document အရေအတွက်

    Returns:
        Retriever instance
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever


def retrieve_with_scores(vectorstore: Chroma, question: str, top_k: int = 3) -> list:
    """
    Similarity search ကို scores နဲ့တကွ ပြုလုပ်ပေးတယ်။

    Args:
        vectorstore: Chroma vector store instance
        question: User query
        top_k: retrieve လုပ်မယ့် document အရေအတွက်

    Returns:
        list: (Document, score) tuples
    """
    return vectorstore.similarity_search_with_score(question, k=top_k)


def format_retrieval_info(docs_with_scores: list) -> str:
    """
    Retrieved docs + scores ကို readable string အဖြစ် format လုပ်ပေးတယ်။
    """
    lines = []
    for rank, (doc, score) in enumerate(docs_with_scores, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        preview = doc.page_content[:300].replace("\n", " ")
        lines.append(
            f"── Chunk #{rank} ──\n"
            f"  Score : {score:.4f} (lower = more similar)\n"
            f"  Page  : {page}\n"
            f"  Source: {source}\n"
            f"  Text  : {preview}...\n"
        )
    return "\n".join(lines)
