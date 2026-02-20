"""
05 - QA Chain
Retriever နဲ့ LLM ကို ပေါင်းစပ်ပြီး RAG chain တည်ဆောက်တယ်။
"""

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


def get_prompt_template() -> ChatPromptTemplate:
    """
    QA chain အတွက် prompt template ကို return ပြန်ပေးတယ်။

    Returns:
        ChatPromptTemplate: Prompt template instance
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    return prompt


def format_augmented_prompt(retrieved_docs: list, user_question: str) -> str:
    """
    Retrieved docs နဲ့ user question ကို ပေါင်းစပ်ပြီး
    LLM ထဲ ဝင်သွားမယ့် augmented prompt ကို preview string ပြန်ပေးတယ်။

    Args:
        retrieved_docs: Retrieved Document objects
        user_question: User ရဲ့ question

    Returns:
        str: Augmented prompt preview
    """
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    filled_system = SYSTEM_PROMPT.replace("{context}", context)

    return (
        f"[SYSTEM]\n{filled_system}\n\n"
        f"[HUMAN]\n{user_question}"
    )


def create_qa_chain(llm, retriever):
    """
    LLM နဲ့ Retriever ကို ပေါင်းစပ်ပြီး RAG chain တည်ဆောက်တယ်။

    Args:
        llm: Language model instance
        retriever: Vector store retriever instance

    Returns:
        RAG chain instance
    """
    prompt = get_prompt_template()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    return qa_chain
