# Golden question set for evaluating final-answer correctness.
#
# Unlike the router eval, this targets ONE specific repo (your own
# github_codebase_assistant), because correctness requires real ground
# truth about that repo's actual content.
#
# Each "key fact" is a short, independently-checkable claim the generated
# answer should reflect. These were drafted from your README.md and
# utils/repo_loader.py — REVIEW THEM before trusting eval results, since
# you're the ground-truth authority on your own project, not me.
#
# Format: (question, [key_fact_1, key_fact_2, ...])

TARGET_REPO_URL = "https://github.com/lakshmicharan18/github_codebase_assistant"

GOLDEN_CORRECTNESS_QUESTIONS = [
    (
        "Who built this project?",
        [
            "Built by Lakshmi Charan Yakkala",
            "The author is a student at RGUKT Nuzvid",
        ],
    ),
    (
        "What problem does this project solve?",
        [
            "Lets users ask natural language questions about any public GitHub repository",
            "Provides accurate, source-grounded answers using LLMs",
        ],
    ),
    (
        "What vector database does this project use?",
        [
            "Uses ChromaDB as the vector database",
        ],
    ),
    (
        "What embedding approach is used for semantic search?",
        [
            "Uses Hugging Face embeddings for semantic search",
        ],
    ),
    (
        "Does this project rerank retrieved documents, and how?",
        [
            "Yes, it reranks retrieved chunks using a cross-encoder reranker",
        ],
    ),
    (
        "What LLM model is used to generate the final answer?",
        [
            "Uses Groq's Llama 3.3 70B Versatile model",
        ],
    ),
    (
        "What is the license of this project?",
        [
            "The project is intended for educational and learning purposes",
        ],
    ),
    (
        "What file types does this project index from a repository?",
        [
            "Supports Python, JavaScript, TypeScript, and Java source files",
            "Also supports Markdown, YAML, JSON, and TOML files",
        ],
    ),
    (
        "Which folders are ignored when scanning a repository?",
        [
            ".git is ignored",
            "node_modules is ignored",
            "venv (or similar virtual environment folders) is ignored",
        ],
    ),
    (
        "Does the project rewrite user questions before retrieving answers?",
        [
            "Yes, it has a question-rewriting step that turns follow-up questions into standalone ones using an LLM",
        ],
    ),
    (
        "Is hybrid search (BM25 + vector search) currently implemented, or planned?",
        [
            "Hybrid search is listed as a future enhancement, not something already implemented",
        ],
    ),
    (
        "How does the system decide which retrieval strategy to use for a question?",
        [
            "An LLM-powered query router classifies the question into a category",
            "The category determines which retrievers/document types are used",
        ],
    ),
]