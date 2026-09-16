"""Small deterministic retrieval hints for project-level authorship questions."""
import re


def is_authorship_question(question):
    text = question.lower()
    project = r'(?:project|repository|repo|codebase|application|app)'
    return bool(
        re.search(r'\bwho\b.*\b(?:built|created|developed|authored|wrote)\b.*\b' + project + r'\b', text)
        or re.search(r'\b(?:author|creator|developer)\b.*\b' + project + r'\b', text)
        or re.search(r'\b' + project + r'\b.*\b(?:author|creator|developer)\b', text)
    )


def retrieval_question(question):
    if is_authorship_question(question):
        return question + ' author creator developed by created by'
    return question


def is_project_overview_question(question):
    text = question.lower()
    return bool(re.search(
        r"\b(?:what problem|what does|what is|purpose|overview|summary|summarize)\b.*\b(?:project|repository|repo|codebase|app|application)\b"
        r"|\b(?:project|repository|repo|codebase|app|application)\b.*\b(?:purpose|overview|summary)\b", text
    ))
