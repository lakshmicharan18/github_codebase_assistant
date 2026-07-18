import os
import re
import shutil
import requests
from git import Repo, GitCommandError
from langchain_core.documents import Document


class RepoValidationError(Exception):
    """Raised when a repo URL fails validation or exceeds allowed limits."""
    pass


# Only allow standard https GitHub repo URLs, e.g.:
#   https://github.com/owner/repo
#   https://github.com/owner/repo.git
#   https://github.com/owner/repo/
GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/([\w.-]+)/([\w.-]+?)(\.git)?/?$"
)

MAX_REPO_SIZE_KB = 150_000  # ~150 MB, GitHub API reports this in KB
CLONE_TIMEOUT_SECONDS = 60


IGNORE_DIRS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".github",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "site-packages",
    "docs/_build"
]

ALLOWED_EXTENSIONS = [
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".html", ".css",
    ".md", ".json", ".yml", ".yaml",
    ".toml", ".txt", ".rst"
]

LOW_VALUE_FILES = [
    "license",
    "license.txt",
    "license.md",
    "copying",
]


def validate_github_url(repo_url):
    """
    Ensures the URL is a well-formed, public https://github.com/<owner>/<repo> URL.
    Raises RepoValidationError otherwise.
    """
    if not repo_url or not isinstance(repo_url, str):
        raise RepoValidationError("Please enter a GitHub repository URL.")

    repo_url = repo_url.strip()
    match = GITHUB_URL_PATTERN.match(repo_url)

    if not match:
        raise RepoValidationError(
            "Only public GitHub repo URLs are supported, "
            "e.g. https://github.com/owner/repo"
        )

    owner, repo, _ = match.groups()
    return owner, repo


def check_repo_size(owner, repo, max_size_kb=MAX_REPO_SIZE_KB):
    """
    Queries the GitHub API for repo metadata before cloning, so we reject
    oversized or non-existent repos without ever running `git clone`.
    Raises RepoValidationError if the repo is missing, private, or too large.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        response = requests.get(api_url, timeout=10)
    except requests.RequestException as e:
        raise RepoValidationError(f"Could not reach GitHub API: {e}")

    if response.status_code == 404:
        raise RepoValidationError(
            "Repository not found. Check the URL and make sure it's public."
        )

    if response.status_code == 403:
        raise RepoValidationError(
            "GitHub API rate limit hit while checking repo size. Try again shortly."
        )

    if response.status_code != 200:
        raise RepoValidationError(
            f"GitHub API returned an unexpected error (status {response.status_code})."
        )

    data = response.json()

    if data.get("private"):
        raise RepoValidationError("Private repositories are not supported yet.")

    size_kb = data.get("size", 0)

    if size_kb > max_size_kb:
        raise RepoValidationError(
            f"Repository is too large to index ({size_kb / 1024:.1f} MB). "
            f"Limit is {max_size_kb / 1024:.0f} MB."
        )

    return size_kb


def clone_github_repo(repo_url, repo_path="repos/cloned_repo"):
    """
    Validates the URL, checks repo size via the GitHub API, then performs
    a shallow (depth=1) clone as defense-in-depth against oversized repos.
    Raises RepoValidationError on any validation failure.
    """
    owner, repo = validate_github_url(repo_url)
    check_repo_size(owner, repo)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    try:
        Repo.clone_from(repo_url, repo_path, depth=1)
    except GitCommandError as e:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        raise RepoValidationError(f"Failed to clone repository: {e}")

    return repo_path


def get_file_type(relative_path):
    path = relative_path.lower()
    file_name = os.path.basename(path)

    if file_name in LOW_VALUE_FILES:
        return "license"

    if file_name in ["requirements.txt", "pyproject.toml", "package.json", "pom.xml"]:
        return "dependency"

    if file_name in ["dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
        return "configuration"

    if file_name in [".env", ".env.example", "config.py", "conf.py"]:
        return "configuration"

    if file_name.startswith("readme"):
        return "readme"

    if "test" in path:
        return "test"

    if path.startswith("docs/") or "/docs/" in path:
        if file_name.endswith((".md", ".rst", ".txt")):
            return "documentation"
        return "configuration"

    if file_name.endswith((".md", ".rst", ".txt")):
        return "documentation"

    return "source_code"


def get_file_priority(relative_path, file_type):
    path = relative_path.lower()
    file_name = os.path.basename(path)

    if file_name == "readme.md":
        return 1
    elif file_type == "readme":
        return 2
    elif file_type == "documentation":
        return 3
    elif path.startswith("src/"):
        return 4
    elif file_type == "source_code":
        return 5
    elif file_type == "configuration":
        return 6
    elif file_type == "dependency":
        return 7
    elif file_type == "test":
        return 8
    elif file_type == "license":
        return 9
    else:
        return 10


def add_line_numbers(content):
    lines = content.splitlines()
    numbered_lines = []

    for i, line in enumerate(lines, start=1):
        numbered_lines.append(f"{i}: {line}")

    return "\n".join(numbered_lines)


def build_repo_structure_document(repo_path):
    structure_lines = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = root.replace(repo_path, "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)

        if folder_name:
            structure_lines.append(f"{indent}{folder_name}/")

        sub_indent = "  " * (level + 1)

        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(tuple(ALLOWED_EXTENSIONS)):
                relative_path = os.path.relpath(file_path, repo_path)
                file_type = get_file_type(relative_path)
                structure_lines.append(f"{sub_indent}{file}  [{file_type}]")

    repo_structure = "\n".join(structure_lines)

    architecture_summary = f"""
Repository Architecture / Project Structure

This document is automatically generated from the repository folder and file structure.
Use this document whenever the user asks about:
- project architecture
- project structure
- folder structure
- high-level design
- codebase organization

Repository tree:

{repo_structure}
"""

    return Document(
        page_content=architecture_summary,
        metadata={
            "source": "AUTO_GENERATED_REPO_STRUCTURE",
            "file_name": "AUTO_GENERATED_REPO_STRUCTURE",
            "file_extension": ".txt",
            "file_type": "repo_structure",
            "file_priority": 0
        }
    )


def load_code_files(repo_path):
    documents = []

    repo_structure_doc = build_repo_structure_document(repo_path)
    documents.append(repo_structure_doc)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(tuple(ALLOWED_EXTENSIONS)):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if not content.strip():
                        continue

                    relative_path = os.path.relpath(file_path, repo_path)
                    file_extension = os.path.splitext(file)[1]
                    file_type = get_file_type(relative_path)
                    file_priority = get_file_priority(relative_path, file_type)

                    content_with_lines = add_line_numbers(content)

                    doc = Document(
                        page_content=content_with_lines,
                        metadata={
                            "source": relative_path,
                            "file_name": file,
                            "file_extension": file_extension,
                            "file_type": file_type,
                            "file_priority": file_priority
                        }
                    )

                    documents.append(doc)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return documents