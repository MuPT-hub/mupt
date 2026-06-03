# Contributing

When contributing to this repository, we recommend starting by making an issue before submitting a pull request.

A fun, welcoming spirit of helping to solve polymer science problems is the primary goal here. 

## Making an Issue
Please open an issue for bug reports and feature requests using the GitHub issue tracker.

If you are reporting a bug, please include the following information:

* Concise description of the bug
* Expected behavior
* Actual behavior
* Steps to reproduce
* Details of your software environment

For feature requests, please include a short description of the feature.

## Submitting a Pull Request
We welcome code contributions in the form of pull requests. Please follow the steps below:

* Fork the repository (default branch is main)
* Create a new branch for your feature/bug fix
* Make your changes
* Commit your changes
* Push your changes to your fork and create a pull request. In your pull request, include a short description and idenitfy any existing To-Do's. Include unit tests in your PR if your code adds new functionality.
* Assign the pull request to a reviewer and wait for the reviewers feedback

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project PI at ericjankowski@boisestate.edu. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage].

[homepage]: http://contributor-covenant.org

## AI Contribution Policy

### Core Philosophy

MuPT accepts AI-assisted code (e.g., using Copilot, Cursor, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision. 

### Principles of Contribution

#### The "Explain It" Standard

 - All pull requests messages and descriptions must be entirely authored by human. Individual commit methods may use AI tools to fully summarize the changes. If the PR is suspected to be AI, the submitters will be asked to rewrite the PR message from scratch. 
 - If a maintainer or reviewer asks during code review, you must be able to derive the math or explain the logic of any function you submit. Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

#### Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

- Every PR introducing functional changes must pass all tests that are currently extant. This is mandatory for all contributors and is particularly important for first-time contributors.
- All code with AI contributions should include tests to demonstrate their validity.  Intially, as the testing framework is developed, this may be relaxed, but will become mandatory in the near future.

#### Hallucination & Redundancy

AI models often hallucinate comments or reinvent existing code.

 - You should use existing `mupt` code and functionality. Creating new helper functions when a MuPT equivalent exists will be rejected
 - "Ghost comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite.

### Transparency in AI Usage Disclosure

All PRs must accurately describe the actual use of AI tools.

**What counts as AI-generated:**
- An AI agent (e.g., Cursor, GitHub Copilot, ChatGPT, etc.) generated the code or commit messages
- You cannot explain the logic without referring to the AI's output
- The PR was created primarily by an agent with minimal human review or modification
- **This should not currently submitted to MuPT**

**When to mark as AI-assisted:**
- You used AI tools for boilerplate code, refactoring, or suggestions, but you manually reviewed and tested every line
- You understand and can explain all the code you're submitting
- You made significant modifications to AI-suggested code
- This code is appropriate for MuPT contribution

**When to mark as No AI used:**
- You wrote all code manually without any AI assistance

Incorrectly marking AI-generated code as "AI-assisted" or "No AI used" may result in PR closure, especially if the code contains hallucinations or cannot be explained during review.

AI Policy concepts drawn from https://github.com/kornia/kornia/AI_POLICY.md
