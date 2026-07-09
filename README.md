💻 GitHub Codebase Assistant

An intelligent Conversational Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about any public GitHub repository. The application clones a repository, indexes its source code and documentation, and provides accurate, source-grounded answers using Large Language Models (LLMs).

This was developed by YAKKALA LAKSHMI CHARAN, a student from RGUKT nuzvid

---

🚀 Features

- Clone and process any public GitHub repository
- Intelligent code and documentation indexing
- Code-aware and documentation-aware chunking
- Hugging Face Embeddings for semantic search
- ChromaDB Vector Database
- LLM-powered Query Routing
- Conversational Question Rewriting
- Multi-Retriever Architecture
- Auto-generated Repository Structure
- Source-aware Answers with File References
- Interactive Streamlit Chat Interface

---

🏗️ System Architecture

                    GitHub Repository URL
                              │
                              ▼
                     Clone Repository
                              │
                              ▼
                    Load Repository Files
                              │
                              ▼
                Generate Repository Structure
                              │
                              ▼
              Code & Documentation Chunking
                              │
                              ▼
                  Generate Vector Embeddings
                              │
                              ▼
                     Store in ChromaDB
                              │
                              ▼
                     User Asks Question
                              │
                              ▼
                 Question Rewriting (LLM)
                              │
                              ▼
                   Query Router (LLM)
                              │
                              ▼
                  Multi-Retriever Search
                              │
             ┌──────────┬──────────┬──────────┐
             │          │          │          │
             ▼          ▼          ▼          ▼
         README      Source     Docs      Repo Structure
        Retriever   Retriever Retriever    Retriever
             │          │          │          │
             └──────────┴──────────┴──────────┘
                              │
                              ▼
                  Merge Retrieved Documents
                              │
                              ▼
                  Remove Duplicate Chunks
                              │
                              ▼
                    Sort by File Priority
                              │
                              ▼
                   Groq LLM Generates Answer
                              │
                              ▼
                  Answer with Source Citations

---

📂 Project Structure

github-codebase-assistant/
│
├── app.py
│
├── utils/
│   ├── repo_loader.py
│   ├── code_splitter.py
│   ├── vector_store.py
│   ├── query_router.py
│   ├── question_rewriter.py
│   ├── multi_retriever.py
│   └── rag_chain.py
│
├── chroma_db/
│
├── repos/
│
├── requirements.txt
│
└── README.md

---

⚙️ Tech Stack

Programming Language

- Python

LLM

- Groq
- Llama 3.3 70B Versatile

Frameworks

- LangChain
- Streamlit

Vector Database

- ChromaDB

Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

Version Control

- Git
- GitHub

---

🔄 Workflow

Step 1

The user enters a public GitHub repository URL.

Example:

https://github.com/pallets/flask

---

Step 2

The repository is cloned locally.

---

Step 3

The application scans all supported files including

- Python
- JavaScript
- TypeScript
- Java
- Markdown
- YAML
- JSON
- TOML

while ignoring unnecessary folders like

- .git
- node_modules
- venv
- build
- dist

---

Step 4

Each file is enriched with metadata such as

- File Name
- File Type
- File Extension
- File Priority

---

Step 5

A repository structure document is automatically generated to improve architecture-related queries.

---

Step 6

Files are split intelligently.

Source code and documentation use different chunking strategies to preserve semantic meaning.

---

Step 7

Each chunk is converted into vector embeddings using Hugging Face Embeddings.

---

Step 8

The embeddings are stored in ChromaDB.

---

Step 9

The user asks a question.

Example:

How are routes implemented?

---

Step 10

If the question is conversational, the Question Rewriter converts it into a standalone question.

Example

Before

Explain that simply.

After

Explain how routes are implemented in simple words.

---

Step 11

The Query Router classifies the question into one of the following categories.

- Overview
- Architecture
- Implementation
- Testing
- Configuration
- Dependency
- License
- General

---

Step 12

The Multi-Retriever retrieves relevant information from multiple sources depending on the question category.

For example,

Architecture questions retrieve from

- Repository Structure
- README
- Documentation

Implementation questions retrieve from

- Source Code
- Documentation

---

Step 13

The retrieved documents are

- Merged
- Deduplicated
- Sorted by priority

before being passed to the LLM.

---

Step 14

The Groq LLM generates a response using only the retrieved repository context and returns source-aware answers.

---

✨ Example Questions

Overview

- What is Flask?
- What problem does this project solve?
- Explain this project.

Architecture

- Explain the project architecture.
- Explain the folder structure.
- What are the main modules?

Implementation

- How are routes implemented?
- Where is Session defined?
- Explain this function.

Configuration

- How is the project configured?
- Where are environment variables defined?

Dependency

- Which libraries are required?
- What packages does this project use?

---

🌟 Key Features

- Conversational RAG
- Semantic Search
- LLM-based Query Routing
- Multi-Retriever Architecture
- Repository Structure Generation
- Source-aware Responses
- Metadata-aware Retrieval
- Duplicate Removal
- Priority-based Context Selection

---

📈 Future Enhancements

- Parent-Child Retrieval
- Hybrid Search (BM25 + Vector Search)
- LLM-based Reranking
- Repository Summarization
- Mermaid Architecture Diagram Generation
- Cross-file Execution Tracing
- Multi-Agent Code Analysis
- Support for Private GitHub Repositories
- Code Review and Bug Detection

---

👨‍💻 Author

Lakshmi Charan Yakkala

- GitHub: https://github.com/lakshmicharan18
- LinkedIn: https://www.linkedin.com/in/charan-yakkala-95bbb8318/

---

📄 License

This project is intended for educational and learning purposes.
