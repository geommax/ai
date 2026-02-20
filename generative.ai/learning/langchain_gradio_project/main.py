"""
RAG QA Bot - Main Entry Point

pip install langchain langchain-huggingface langchain-chroma langchain-community \
            chromadb gradio pypdf sentence-transformers langchain-classic \
            pytesseract Pillow python-docx openpyxl unstructured
"""

import importlib
import gradio as gr

# Numeric-prefixed filenames များကို importlib နဲ့ import လုပ်ရတယ်
_load_sources     = importlib.import_module("01_load_sources")
_ingestion        = importlib.import_module("02_ingestion")
_vector_embedding = importlib.import_module("03_vector_embedding")
_chromadb_mgmt    = importlib.import_module("04_chromadb_managements")
_retrieval        = importlib.import_module("06_retrieval")
_chain            = importlib.import_module("05_chain")
_llm_module       = importlib.import_module("07_llm")
_gradio_ui        = importlib.import_module("08_gradio")

load_file              = _load_sources.load_file
split_documents        = _ingestion.split_documents
get_embedding_model    = _vector_embedding.get_embedding_model
create_vectorstore     = _chromadb_mgmt.create_vectorstore
get_retriever          = _retrieval.get_retriever
retrieve_with_scores   = _retrieval.retrieve_with_scores
format_retrieval_info  = _retrieval.format_retrieval_info
create_qa_chain        = _chain.create_qa_chain
format_augmented_prompt = _chain.format_augmented_prompt
load_llm               = _llm_module.load_llm
build_interface        = _gradio_ui.build_interface

# ── Initialize models (loaded once at startup) ──────────────────────────
llm = load_llm()
embedding_model = get_embedding_model()

# ── Global state ─────────────────────────────────────────────────────────
qa_chain = None
vectorstore = None


def process_document(uploaded_file):
    """Document ကို load, split, embed, index လုပ်ပြီး QA chain ကို တည်ဆောက်တယ်။"""
    global qa_chain, vectorstore

    if uploaded_file is None:
        return "Please upload a file first."

    # 1. Load (auto-detect file type)
    try:
        docs = load_file(uploaded_file)
    except ValueError as e:
        return str(e)

    if not docs:
        return "No text could be extracted from the file."

    # 2. Split
    splits = split_documents(docs)

    # 3. Store in ChromaDB
    vectorstore = create_vectorstore(splits, embedding_model)

    # 4. Retriever
    retriever = get_retriever(vectorstore)

    # 5. Build QA chain
    qa_chain = create_qa_chain(llm, retriever)

    return f"Processed {len(splits)} chunks from the document. You can now ask questions!"


def answer_question(question):
    """User ရဲ့ question ကို QA chain သုံးပြီး answer ပြန်ပေးတယ်။"""
    if qa_chain is None:
        return "Please upload and process a document first.", "", ""

    # 1. Retrieve with scores (for debug display)
    docs_with_scores = retrieve_with_scores(vectorstore, question)
    retrieval_info = format_retrieval_info(docs_with_scores)

    # 2. Build augmented prompt preview
    retrieved_docs = [doc for doc, _score in docs_with_scores]
    augmented_prompt = format_augmented_prompt(retrieved_docs, question)

    # 3. Run QA chain
    response = qa_chain.invoke({"input": question})

    return response["answer"], retrieval_info, augmented_prompt


if __name__ == "__main__":
    demo = build_interface(process_document, answer_question)
    demo.launch(theme=gr.themes.Soft())
