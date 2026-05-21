# MuPT AI & Authorship Policy

**Version:** 1.0
**Applicability:** All Pull Requests (Human & Bot)

## Core Philosophy

MuPT accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision.

## Principles of Contribution

### Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

- Every PR introducing functional changes must pass all tests that are currently extant. This is mandatory for all contributors and is particularly important for first-time contributors.
- All code with AI contributions should include tests to demonstrate their validity.  Intially, as the testing framework develops, this may be relaxes, but will become mandatory in the near future.

### Hallucination & Redundancy

AI models often hallucinate comments or reinvent existing utilities.

 - You should  existing `mupt` utilities and never reinvent the wheel, except for when the utility is not available. Creating new helper functions when a MuPT equivalent exists will be rejected
 - "Ghost comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite.

### The "Explain It" Standard

 - All pull requests messages and descriptions must be entirely authored by human. Individual commit methods may use AI tools to fully summarize the changes. If the PR is suspected to be AI, the submitters will be asked to rewrite the PR message from scratch. 
 - If a maintainer or reviewer asks during code review, you must be able to derive the math or explain the logic of any function you submit. Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

### Transparency in AI Usage Disclosure

All PRs must accurately describe the actual use of AI tools.

**When to mark as 🔴 AI-generated:**
- An AI agent (e.g., Cursor, GitHub Copilot, ChatGPT, etc.) generated the code or commit messages
- You cannot explain the logic without referring to the AI's output
- The PR was created primarily by an agent with minimal human review or modification
- This should not currently submitted to MuPT

**When to mark as 🟡 AI-assisted:**
- You used AI tools for boilerplate code, refactoring, or suggestions, but you manually reviewed and tested every line
- You understand and can explain all the code you're submitting
- You made significant modifications to AI-suggested code
- This code is appropriate for MuPT contribution

**When to mark as 🟢 No AI used:**
- You wrote all code manually without any AI assistance

Incorrectly marking AI-generated code as "AI-assisted" or "No AI used" may result in PR closure, especially if the code contains hallucinations or cannot be explained during review.

This document inspired by policies developed at https://github.com/kornia/kornia/AI_POLICY.md
