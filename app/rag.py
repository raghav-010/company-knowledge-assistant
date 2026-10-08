from __future__ import annotations
from typing import List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from .utils import get_vector_store

SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Question:\n{input}\n\n"
     "Context:\n{context}\n\n"
     "Rule: Prefer the most recent policy by effective date.")
])

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

async def _build_chain(category: str | None = None):
    store = await get_vector_store()
    search_kwargs = {"k": int(os.getenv("RETRIEVAL_K", "5"))}
    if category:
        search_kwargs["filter"] = {"category": category}
    base_retriever = store.as_retriever(search_kwargs=search_kwargs)
    compressor = CohereRerank(top_n=3, model="rerank-multilingual-v3.0")
    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.8-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    chain = (
        {"context": retriever | _format_docs, "input": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain

async def answer_with_docs_async(
    question: str,
    category: str | None = None,
) -> Tuple[str, List[str], List[str]]:
    retriever, chain = await _build_chain(category)
    docs = await retriever.ainvoke(question)
    answer = await chain.ainvoke(question)
    unique_sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
    sources = sorted(unique_sources)
    contexts = [d.page_content for d in docs]
    return answer, sources, contexts