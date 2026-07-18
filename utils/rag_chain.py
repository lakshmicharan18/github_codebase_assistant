from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.query_router import route_question
from utils.multi_retriever import multi_retrieve
from utils.reranker import get_reranker

load_dotenv()

def format_documents(docs):
    context_text = ""

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        file_type = doc.metadata.get("file_type", "unknown")
        priority = doc.metadata.get("file_priority", "unknown")

        context_text += f"\n\n--- Source {i} ---\n"
        context_text += f"File: {source}\n"
        context_text += f"Type: {file_type}\n"
        context_text += f"Priority: {priority}\n"
        context_text += "Content:\n"
        context_text += doc.page_content

    return context_text


def create_rag_chain(vector_db, groq_api_key, question, chat_history_text=""):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    category = route_question(question, groq_api_key)
    
    candidate_docs = multi_retrieve(
        vector_db=vector_db,
        question=question,
        category=category
    )

    reranker = get_reranker()
    retrieved_docs = reranker.rerank(
        question=question,
        documents=candidate_docs,
        top_k = 6
    )

    context_text = format_documents(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a GitHub Codebase Assistant.

        Question category:
        {category}

        Recent conversation:
        {chat_history}

        Use only the retrieved codebase context to answer.

        Important rules:
        1. Explain in simple language.
        2. Always mention exact file paths from the context.
        3. If line numbers are visible in the context, mention them when useful.
        4. If the question is about architecture, prioritize AUTO_GENERATED_REPO_STRUCTURE.
        5. If the question is about what the project is, prioritize README.
        6. If the retrieved context clearly answers the question, answer directly without saying it is partial.
Only say "partial answer" when important requested details are missing.
        7. If the answer is not present in the retrieved context, say:
           "I don't know from this codebase."
        8. Do not make assumptions outside the retrieved context.

        Retrieved context:
        {context}

        User question:
        {input}

        Answer:
        """
    )

    chain = prompt | llm

    return chain, category, retrieved_docs, context_text














'''from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()


def create_rag_chain(vector_db, groq_api_key, question):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    question_lower = question.lower()

    if question_lower.startswith("what is") or "overview" in question_lower or "explain project" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {
                    "$or": [
                        {"file_type": "readme"},
                        {"file_type": "docs"},
                        {"file_type": "documentation"}
                    ]
                }
            }
        )
    elif "license" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"file_name": "LICENSE.txt"}
            }
        )
    elif "test" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"file_type": "test"}
            }
        )
    else:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 6,
                "filter": {
                    "$or": [
                        {"file_type": "source_code"},
                        {"file_type": "readme"},
                        {"file_type": "docs"}
                    ]
                }
            }
        )
        
        

    prompt = ChatPromptTemplate.from_template(
        """
        You are a GitHub Codebase Assistant.

        Answer the user's question using only the given codebase context.

        Rules:
        1. Explain in simple language.
        2. Always mention file paths from the context.
        3. If the answer is found in README or docs, explain it as project-level information.
        4. If the answer is found in source code, explain it as implementation-level information.
        5. If the answer is not present in the context, say:
           "I don't know from this codebase."
        6. Do not make assumptions outside the codebase.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain'''















'''from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()


def create_rag_chain(vector_db, groq_api_key, question):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    question_lower = question.lower()

    if question_lower.startswith("what is") or "overview" in question_lower or "explain project" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {
                    "$or": [
                        {"file_type": "readme"},
                        {"file_type": "docs"},
                        {"file_type": "documentation"}
                    ]
                }
            }
        )
    elif "license" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"file_name": "LICENSE.txt"}
            }
        )
    elif "test" in question_lower:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"file_type": "test"}
            }
        )
    else:
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 6,
                "filter": {
                    "$or": [
                        {"file_type": "source_code"},
                        {"file_type": "readme"},
                        {"file_type": "docs"}
                    ]
                }
            }
        )
        
        

    prompt = ChatPromptTemplate.from_template(
        """
        You are a GitHub Codebase Assistant.

        Answer the user's question using only the given codebase context.

        Rules:
        1. Explain in simple language.
        2. Always mention file paths from the context.
        3. If the answer is found in README or docs, explain it as project-level information.
        4. If the answer is found in source code, explain it as implementation-level information.
        5. If the answer is not present in the context, say:
           "I don't know from this codebase."
        6. Do not make assumptions outside the codebase.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain'''