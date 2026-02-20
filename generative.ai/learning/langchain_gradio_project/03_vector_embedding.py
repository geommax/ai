"""
03 - Vector Embedding Model
Text ကို vector embeddings အဖြစ် ပြောင်းပေးတဲ့ embedding model ကို initialize လုပ်တယ်။
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only: bool = True,
) -> HuggingFaceEmbeddings:
    """
    HuggingFace Embedding model ကို initialize လုပ်ပြီး return ပြန်ပေးတယ်။

    Args:
        model_name: HuggingFace embedding model name
        local_files_only: True ဆိုရင် local cache ကပဲ load လုပ်မယ်

    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"local_files_only": local_files_only},
    )
    return embedding_model
