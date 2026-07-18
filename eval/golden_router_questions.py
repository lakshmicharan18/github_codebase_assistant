# Golden question set for evaluating utils/query_router.py's route_question().
#
# The router only looks at the question text (it doesn't see repo content),
# so this set is repo-agnostic and can be reused across any repo you test.
#
# Each entry: (question, expected_category)
#
# Mix of "easy" (obvious keyword match) and "tricky" (ambiguous phrasing,
# no obvious keyword, or could plausibly fit two categories) questions per
# category — the easy ones tell you the router works at all, the tricky
# ones tell you if it actually generalizes.

GOLDEN_QUESTIONS = [
    # ---------- overview ----------
    ("What is this project?", "overview"),
    ("Give me a summary of what this repo does.", "overview"),
    ("What problem does this project solve?", "overview"),
    ("Can you explain this codebase in simple terms?", "overview"),
    ("Why would someone use this?", "overview"),

    # ---------- architecture ----------
    ("Explain the project architecture.", "architecture"),
    ("What are the main folders and what do they contain?", "architecture"),
    ("How is the codebase organized?", "architecture"),
    ("What are the major modules and how do they connect?", "architecture"),
    ("Walk me through the high-level design of this system.", "architecture"),

    # ---------- implementation ----------
    ("How are routes implemented in this project?", "implementation"),
    ("Where is the main function defined?", "implementation"),
    ("Explain how the retry logic works.", "implementation"),
    ("What does the process_data function do?", "implementation"),
    ("How is authentication handled in the code?", "implementation"),

    # ---------- testing ----------
    ("Are there any unit tests for this?", "testing"),
    ("How do I run the test suite?", "testing"),
    ("What testing framework is used?", "testing"),
    ("Is there test coverage for the API endpoints?", "testing"),
    ("Where are the pytest fixtures defined?", "testing"),

    # ---------- configuration ----------
    ("How do I configure this project?", "configuration"),
    ("What environment variables does this need?", "configuration"),
    ("Is there a Docker setup?", "configuration"),
    ("Where are the settings defined?", "configuration"),
    ("How do I set this up locally?", "configuration"),

    # ---------- dependency ----------
    ("What libraries does this project use?", "dependency"),
    ("What version of Python is required?", "dependency"),
    ("What are the dependencies listed in requirements.txt?", "dependency"),
    ("Does this use any external APIs or SDKs?", "dependency"),
    ("What packages need to be installed?", "dependency"),

    # ---------- license ----------
    ("What license does this project use?", "license"),
    ("Can I use this code commercially?", "license"),
    ("Is this open source?", "license"),
    ("What are the usage restrictions on this repo?", "license"),
    ("Am I allowed to redistribute this code?", "license"),

    # ---------- general / tricky ----------
    ("recheck that", "general"),  # follow-up style, no real content
    ("thanks, that's helpful", "general"),
    ("What's the weather like today?", "general"),  # off-topic
    ("Can you write me a poem about this repo?", "general"),
    ("hi", "general"),

    # ---------- deliberately ambiguous (could plausibly span categories) ----------
    ("How does the project structure affect testing strategy?", "architecture"),
    ("Which config file controls the test environment?", "configuration"),
    ("What does the README say about installation?", "overview"),
    ("Is pytest a dependency or a dev tool only?", "dependency"),
    ("Explain how the Docker config ties into the folder layout.", "configuration"),
]