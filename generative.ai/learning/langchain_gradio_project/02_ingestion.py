"""
02 - Document Ingestion (Text Splitting)
Document တွေကို chunk တွေအဖြစ် ခွဲထုတ်ပေးတယ်။
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Document list ကို သေးငယ်တဲ့ chunk တွေအဖြစ် ခွဲထုတ်ပေးတယ်။
    Splitting text helps the model digest large documents in chunks.

    Args:
        docs: LangChain Document objects များ
        chunk_size: chunk တစ်ခုရဲ့ အရွယ်အစား (characters)
        chunk_overlap: chunk တွေကြား ထပ်နေတဲ့ characters အရေအတွက်

    Returns:
        list: ခွဲထုတ်ထားတဲ့ Document chunks များ
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = text_splitter.split_documents(docs)
    return splits
