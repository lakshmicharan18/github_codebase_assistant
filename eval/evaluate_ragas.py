"""
Evaluates retrieval and generation quality SEPARATELY using the RAGAS
framework, closing a gap the other two eval scripts don't cover:
evaluate_correctness.py only scores the final answer, so a low score there
doesn't tell you whether retrieval fetched the wrong chunks or generation
misused the right ones. This script scores each stage on its own:

    - Context Precision : are the retrieved chunks actually relevant to the
                           question, with the relevant ones ranked higher?
                           (retrieval quality)
    - Faithfulness      : does the generated answer only state things
                           actually supported by the retrieved chunks, or
                           does it hallucinate beyond them?
                           (generation quality)
    - Response Relevancy: does the generated answer actually address the
                           question asked, rather than going off-topic?
                           (generation quality)

All three metrics are reference-free (no hand-written ground truth needed),
so this reuses the same golden questions as evaluate_correctness.py — only
the questions, not the key_facts, since those aren't needed here.

IMPORTANT — like the other eval scripts, this is completely standalone:
    - Never imported by app.py, has ZERO effect on live app behavior.
    - Clones its own throwaway copy of the target repo and deletes its
      vector store when done.

Usage:
    python -m eval.evaluate_ragas

Requires GROQ_API_KEY in your environment/.env (used both for running the
real pipeline AND as the judge LLM for RAGAS's metrics — no OpenAI key
needed, since RAGAS's default OpenAI judge is swapped out for ChatGroq
below).
"""

import csv
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.run_config import RunConfig

from utils.repo_loader import clone_github_repo, load_code_files, RepoValidationError
from utils.code_splitter import split_code_files
from utils.vector_store import create_vector_store, delete_vector_store, get_embedding_model
from utils.rag_chain import create_rag_chain

from eval.golden_correctness_questions import GOLDEN_CORRECTNESS_QUESTIONS, TARGET_REPO_URL

load_dotenv()

EVAL_SESSION_ID = "ragas_eval"
EVAL_REPO_PATH = os.path.join("repos", EVAL_SESSION_ID, "cloned_repo")


def setup_pipeline():
    """Clones the target repo and builds a throwaway vector store for this eval run."""
    print(f"Setting up pipeline for: {TARGET_REPO_URL}")

    repo_path = clone_github_repo(TARGET_REPO_URL, repo_path=EVAL_REPO_PATH)
    documents = load_code_files(repo_path)
    chunks = split_code_files(documents)
    vector_db = create_vector_store(chunks, session_id=EVAL_SESSION_ID)

    print(f"Indexed {len(documents)} files into {len(chunks)} chunks.\n")
    return vector_db


def format_context_for_ragas(retrieved_docs):
    """
    Mirrors utils.rag_chain.format_documents()'s per-chunk formatting
    ("--- Source N ---", File/Type/Priority header, then content), but as a
    list of separate strings rather than one combined block, since RAGAS
    expects retrieved_contexts as a list.

    This matters because the generation prompt shows the LLM file paths and
    "Source N" labels, and its answers cite them accordingly (as instructed
    by the prompt). If RAGAS is instead handed bare doc.page_content with
    no file/source labels, it has no way to verify those citations and
    marks them as unsupported -- an artifact of the eval harness showing
    the judge less than the LLM actually saw, not real hallucination.
    """
    formatted = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        file_type = doc.metadata.get("file_type", "unknown")
        priority = doc.metadata.get("file_priority", "unknown")

        formatted.append(
            f"--- Source {i} ---\n"
            f"File: {source}\n"
            f"Type: {file_type}\n"
            f"Priority: {priority}\n"
            f"Content:\n{doc.page_content}"
        )

    return formatted


def run_pipeline_for_question(vector_db, groq_api_key, question):
    """
    Runs the real pipeline exactly the way app.py does, minus the Streamlit UI.
    Returns the question, the exact chunks fed to the LLM (post-rerank, the
    same ones the user sees in "Source Code Chunks Used"), formatted the same
    way the LLM saw them, and the generated answer.
    """
    chain, category, retrieved_docs, context_text = create_rag_chain(
        vector_db, groq_api_key, question, chat_history_text=""
    )

    response = chain.invoke({
        "input": question,
        "category": category,
        "chat_history": "",
        "context": context_text
    })

    retrieved_contexts = format_context_for_ragas(retrieved_docs)

    return retrieved_contexts, response.content, category


def run_eval():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in your .env file or environment.")
        sys.exit(1)

    try:
        vector_db = setup_pipeline()
    except RepoValidationError as e:
        print(f"ERROR setting up pipeline: {e}")
        sys.exit(1)

    questions = [q for q, _key_facts in GOLDEN_CORRECTNESS_QUESTIONS]

    print(f"Running pipeline on {len(questions)} questions to build the RAGAS dataset...\n")

    samples = []
    categories = []

    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {question}")

        retrieved_contexts, answer, category = run_pipeline_for_question(
            vector_db, groq_api_key, question
        )
        categories.append(category)

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            response=answer
        ))

    # --- Cleanup throwaway resources ---
    delete_vector_store(EVAL_SESSION_ID)

    # RAGAS's judge LLM and embeddings default to OpenAI — swap in Groq and
    # HuggingFace embeddings so this eval only needs the GROQ_API_KEY you
    # already have configured. Deliberately a *different, smaller* model
    # than the app's own llama-3.3-70b-versatile: RAGAS's judge tasks (claim
    # decomposition, relevance checks) don't need a 70B model, and using a
    # separate model means this eval draws from its own daily token quota
    # instead of competing with (and potentially exhausting) whatever budget
    # the production pipeline has already used today.
    judge_llm = LangchainLLMWrapper(ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    dataset = EvaluationDataset(samples=samples)

    print("\nScoring with RAGAS (Context Precision, Faithfulness, Response Relevancy)...\n")

    # RAGAS defaults to 16 concurrent judge calls (max_workers), which
    # overwhelms Groq's rate limits. Faithfulness needs multiple sequential
    # LLM calls per sample (claim decomposition, then verification), so it
    # kept timing out even at max_workers=3 -- Groq's per-minute rate limit
    # forces retries that eat into the timeout budget once several of
    # Faithfulness's extra calls are in flight at once. Fully serial
    # (max_workers=1) removes that contention entirely: slower overall, but
    # every job gets a real score instead of some silently landing as NaN.
    run_config = RunConfig(max_workers=1, timeout=180)

    result = evaluate(
        dataset=dataset,
        metrics=[
            LLMContextPrecisionWithoutReference(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    scores_df = result.to_pandas()
    scores_df.insert(1, "category", categories)

    # --- Write results to CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), "ragas_eval_results.csv")
    scores_df.to_csv(csv_path, index=False)

    # --- Summary ---
    precision_col = "llm_context_precision_without_reference"
    faithfulness_col = "faithfulness"
    relevancy_col = "answer_relevancy"

    avg_precision = scores_df[precision_col].mean()
    avg_faithfulness = scores_df[faithfulness_col].mean()
    avg_relevancy = scores_df[relevancy_col].mean()

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Context Precision (retrieval quality) : {avg_precision:.3f}")
    print(f"Faithfulness       (generation quality): {avg_faithfulness:.3f}")
    print(f"Response Relevancy (generation quality): {avg_relevancy:.3f}")
    print("=" * 60)

    print(
        "\nHow to read a low score:\n"
        "  - Low Context Precision -> retrieval fetched irrelevant chunks or ranked\n"
        "    the relevant ones too low. Look at utils/multi_retriever.py and\n"
        "    utils/reranker.py for that question's category.\n"
        "  - Low Faithfulness -> the LLM stated things not supported by the\n"
        "    retrieved chunks (hallucination), even if retrieval itself was fine.\n"
        "  - Low Response Relevancy -> the answer drifted off-topic from the\n"
        "    actual question, independent of whether the context was correct."
    )

    print(f"\nFull per-question results saved to: {csv_path}")

    return scores_df


if __name__ == "__main__":
    run_eval()
