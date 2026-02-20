"""
04 - ChromaDB Vector Store Management
Document chunks တွေကို ChromaDB ထဲ store လုပ်ပြီး manage လုပ်တယ်။
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(
    splits: list,
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str = "rag_collection",
) -> Chroma:
    """
    Document chunks တွေကို embedding လုပ်ပြီး ChromaDB vectorstore ထဲ store လုပ်တယ်။

    Args:
        splits: ခွဲထုတ်ထားတဲ့ Document chunks များ
        embedding_model: Embedding model instance
        collection_name: ChromaDB collection name

    Returns:
        Chroma: Vector store instance
    """
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=collection_name,
    )
    return vectorstore
