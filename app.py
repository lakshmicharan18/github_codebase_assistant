import streamlit as st
from dotenv import load_dotenv

from utils.repo_loader import clone_github_repo, load_code_files
from utils.code_splitter import split_code_files
from utils.vector_store import create_vector_store
from utils.rag_chain import create_rag_chain
from utils.question_rewriter import rewrite_question

load_dotenv()

st.set_page_config(page_title="GitHub Codebase Assistant", page_icon="💻")

st.title("💻 GitHub Codebase Assistant")
st.write("Enter a GitHub repository URL and ask questions about the codebase.")

groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
repo_url = st.text_input("Enter GitHub Repo URL")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if st.button("Process Repository"):
    if not repo_url:
        st.warning("Please enter a GitHub repo URL")
    else:
        with st.spinner("Processing repository..."):
            repo_path = clone_github_repo(repo_url)
            documents = load_code_files(repo_path)
            chunks = split_code_files(documents)
            vector_db = create_vector_store(chunks)

            st.session_state.vector_db = vector_db
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.repo_url = repo_url
            st.session_state.messages = []

        st.success("Repository indexed successfully!")
        st.write("Total files loaded:", len(documents))
        st.write("Total chunks created:", len(chunks))

        with st.expander("View loaded files"):
            for doc in documents[:30]:
                st.write(
                    doc.metadata.get("source"),
                    "|",
                    doc.metadata.get("file_type")
                )

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_question = st.chat_input("Ask question about the codebase")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    st.chat_message("user").write(user_question)

    if "vector_db" not in st.session_state:
        st.warning("Please process a repository first.")
    elif not groq_api_key:
        st.warning("Please enter Groq API Key.")
    else:
        with st.spinner("Generating answer..."):
            standalone_question = rewrite_question(
                user_question,
                st.session_state.messages,
                groq_api_key
            )

            chat_history_text = ""

            for msg in st.session_state.messages[-6:]:
                chat_history_text += f"{msg['role']}: {msg['content']}\n"

            chain, category, retrieved_docs, context_text = create_rag_chain(
                st.session_state.vector_db,
                groq_api_key,
                standalone_question,
                chat_history_text
            )

            response = chain.invoke({
                "input": standalone_question,
                "category": category,
                "chat_history": chat_history_text,
                "context": context_text
            })

            answer = response.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)

            st.write("### Question Category")
            st.info(category)

            st.write("### Rewritten Question")
            st.info(standalone_question)

            with st.expander("Source Code Chunks Used"):
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"### Source {i + 1}")
                    st.write("📄 File:", doc.metadata.get("source"))
                    st.write("📂 Type:", doc.metadata.get("file_type"))
                    st.write("⭐ Priority:", doc.metadata.get("file_priority"))
                    st.write("Reranker Score:", round(doc.metadata.get("reranker_score", 0), 4))
                    st.code(doc.page_content[:1200])
                    st.write("--------")