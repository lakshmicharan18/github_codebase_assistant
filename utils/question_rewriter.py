from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rewrite_question(question, chat_history, groq_api_key):
    if not chat_history or len(chat_history) <= 1:
        return question

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    history_text = ""

    for msg in chat_history[-6:]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = ChatPromptTemplate.from_template(
        """
        You are a question rewriting assistant for a GitHub Codebase Assistant.

        Your task:
        Rewrite the latest user question into a standalone question only when needed.

        Rules:
        1. If the latest question is already clear, return it unchanged.
        2. If the latest question uses words like "that", "this", "it", "there", or "recheck",
           rewrite it using the recent conversation context.
        3. Do not change the user's intent.
        4. Do not add extra assumptions.
        5. Do not answer the question.
        6. Return only the rewritten question.

        Chat history:
        {chat_history}

        Latest user question:
        {question}

        Standalone question:
        """
    )

    chain = prompt | llm | StrOutputParser()

    rewritten_question = chain.invoke({
        "chat_history": history_text,
        "question": question
    })

    return rewritten_question.strip()