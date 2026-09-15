import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


VALID_CATEGORIES = [
    "overview",
    "architecture",
    "implementation",
    "testing",
    "configuration",
    "dependency",
    "license",
    "general"
]


def route_question(question, groq_api_key):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a query router for a GitHub Codebase Assistant.

        Classify the user question into exactly one category.

        Categories:

        overview:
        Use when the user asks what the project is, what it does, summary,
        purpose, introduction, or high-level explanation.

        architecture:
        Use when the user asks about project architecture, folder structure,
        codebase organization, modules, high-level design, or how the repo is organized.

        implementation:
        Use when the user asks about source code logic, functions, classes,
        routes, APIs, workflows, business logic, or how something is implemented.

        testing:
        Use when the user asks about tests, test cases, unit tests, pytest,
        test files, or testing strategy.

        configuration:
        Use when the user asks about config files, environment variables,
        settings, Docker, YAML, TOML, JSON config, or setup configuration.

        dependency:
        Use when the user asks about packages, libraries, requirements,
        dependencies, versions, pyproject, package.json, or pom.xml.

        license:
        Use when the user asks about license, copyright, usage permission,
        redistribution, or legal usage.

        general:
        Use only when the question does not fit any category above.

        User question:
        {question}

        Return only one category name.
        Do not explain.
        """
    )

    chain = prompt | llm | StrOutputParser()

    category = chain.invoke({"question": question})
    category = category.strip().lower()

    if category not in VALID_CATEGORIES:
        category = "general"

    return category