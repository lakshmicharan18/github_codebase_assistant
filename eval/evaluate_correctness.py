"""
Evaluates final-answer CORRECTNESS by running your actual RAG pipeline
(clone -> index -> retrieve -> rerank -> generate) against a golden set
of questions, then using a separate LLM judge call to check whether each
generated answer actually contains the required key facts.

IMPORTANT — this is a completely offline, standalone script:
    - It is never imported by app.py and never runs during a real user's
      chat session. It has ZERO effect on live app latency.
    - It clones its own throwaway copy of the target repo and builds its
      own throwaway vector store, isolated from anything a real user does.

Usage:
    python -m eval.evaluate_correctness

Requires GROQ_API_KEY in your environment/.env (used both for generating
answers via your real pipeline AND for the separate judge calls).
"""

import csv
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.repo_loader import clone_github_repo, load_code_files, RepoValidationError
from utils.code_splitter import split_code_files
from utils.vector_store import create_vector_store, delete_vector_store
from utils.rag_chain import create_rag_chain

from eval.golden_correctness_questions import GOLDEN_CORRECTNESS_QUESTIONS, TARGET_REPO_URL

load_dotenv()

EVAL_SESSION_ID = "correctness_eval"
EVAL_REPO_PATH = os.path.join("repos", EVAL_SESSION_ID, "cloned_repo")

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """
You are evaluating whether a generated answer correctly covers a list of required facts.
Judge only the CONTENT of the answer against each fact — ignore tone, formatting, or length.

Question:
{question}

Generated Answer:
{answer}

Required Facts:
{numbered_facts}

For EACH fact, classify it as exactly one of:
- "present": the answer clearly states this fact, or an equivalent in different words
- "absent": the answer does not mention this fact at all
- "contradicted": the answer states something that conflicts with this fact

Respond with ONLY valid JSON, no other text, no markdown code fences, in this exact format:
{{"results": [{{"fact": "<fact text>", "status": "present"}}, ...]}}
"""
)


def setup_pipeline(groq_api_key):
    """Clones the target repo and builds a throwaway vector store for this eval run."""
    print(f"Setting up pipeline for: {TARGET_REPO_URL}")

    repo_path = clone_github_repo(TARGET_REPO_URL, repo_path=EVAL_REPO_PATH)
    documents = load_code_files(repo_path)
    chunks = split_code_files(documents)
    vector_db = create_vector_store(chunks, session_id=EVAL_SESSION_ID)

    print(f"Indexed {len(documents)} files into {len(chunks)} chunks.\n")
    return vector_db


def generate_answer(vector_db, groq_api_key, question):
    """Runs the real pipeline exactly the way app.py does, minus the Streamlit UI."""
    chain, category, retrieved_docs, context_text = create_rag_chain(
        vector_db, groq_api_key, question, chat_history_text=""
    )

    response = chain.invoke({
        "input": question,
        "category": category,
        "chat_history": "",
        "context": context_text
    })

    return response.content, category


def judge_correctness(question, answer, key_facts, groq_api_key):
    """
    Separate LLM call (not part of the generation pipeline) that checks
    each key fact against the generated answer.
    Returns a list of {"fact": ..., "status": "present"|"absent"|"contradicted"}.
    Falls back to marking all facts "absent" if judging fails, so a
    parsing error shows up as a low score rather than crashing the run.
    """
    judge_llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0
    )

    numbered_facts = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(key_facts))

    chain = JUDGE_PROMPT | judge_llm

    try:
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "numbered_facts": numbered_facts
        })

        raw = response.content.strip()
        # Defensive: strip markdown code fences if the model adds them anyway
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        import json
        parsed = json.loads(raw)
        return parsed["results"]

    except Exception as e:
        print(f"  [judge error: {e}] — marking all facts as absent for this question")
        return [{"fact": fact, "status": "absent"} for fact in key_facts]


def run_eval():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in your .env file or environment.")
        sys.exit(1)

    try:
        vector_db = setup_pipeline(groq_api_key)
    except RepoValidationError as e:
        print(f"ERROR setting up pipeline: {e}")
        sys.exit(1)

    all_results = []
    total_facts = 0
    total_present = 0
    total_contradicted = 0

    print(f"Running correctness evaluation on {len(GOLDEN_CORRECTNESS_QUESTIONS)} questions...\n")

    for i, (question, key_facts) in enumerate(GOLDEN_CORRECTNESS_QUESTIONS, start=1):
        print(f"[{i}/{len(GOLDEN_CORRECTNESS_QUESTIONS)}] {question}")

        answer, category = generate_answer(vector_db, groq_api_key, question)
        fact_results = judge_correctness(question, answer, key_facts, groq_api_key)

        present = sum(1 for r in fact_results if r["status"] == "present")
        contradicted = sum(1 for r in fact_results if r["status"] == "contradicted")
        score = present / len(key_facts) if key_facts else 0

        total_facts += len(key_facts)
        total_present += present
        total_contradicted += contradicted

        print(f"    Score: {present}/{len(key_facts)} facts present"
              + (f", {contradicted} CONTRADICTED" if contradicted else ""))

        all_results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "score": f"{present}/{len(key_facts)}",
            "contradicted_count": contradicted,
            "fact_details": fact_results
        })

        time.sleep(0.3)

    # --- Cleanup throwaway resources ---
    delete_vector_store(EVAL_SESSION_ID)

    # --- Write results to CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), "correctness_eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "category", "score", "contradicted_count", "answer"])
        for r in all_results:
            writer.writerow([r["question"], r["category"], r["score"], r["contradicted_count"], r["answer"]])

    # --- Summary ---
    overall_pct = total_present / total_facts * 100 if total_facts else 0

    print("\n" + "=" * 60)
    print(f"OVERALL CORRECTNESS: {total_present}/{total_facts} facts present ({overall_pct:.1f}%)")
    print(f"CONTRADICTIONS FOUND: {total_contradicted}")
    print("=" * 60)

    if total_contradicted > 0:
        print("\n⚠️  Contradictions are worse than missing facts — the model stated")
        print("something factually wrong, not just incomplete. Review these first:")
        for r in all_results:
            if r["contradicted_count"] > 0:
                print(f"\n  Q: {r['question']}")
                for fd in r["fact_details"]:
                    if fd["status"] == "contradicted":
                        print(f"    CONTRADICTED fact: {fd['fact']}")
                print(f"    Answer given: {r['answer'][:200]}...")

    low_scoring = [r for r in all_results if r["score"].split("/")[0] != r["score"].split("/")[1]]
    if low_scoring:
        print(f"\nQuestions with missing facts ({len(low_scoring)}):")
        for r in low_scoring:
            print(f"  '{r['question']}' -> {r['score']}")

    print(f"\nFull results saved to: {csv_path}")

    if overall_pct < 80:
        print(
            "\n⚠️  Correctness is below 80% — check whether this is a retrieval "
            "problem (right chunks not being found) or a generation problem "
            "(right chunks found, but answer still misses/contradicts facts). "
            "Compare against the 'Source Code Chunks Used' in the app for the "
            "failing questions to tell which stage is at fault."
        )

    return overall_pct, all_results


if __name__ == "__main__":
    run_eval()