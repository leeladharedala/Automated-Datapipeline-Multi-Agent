"""
submit_pr tool — creates a GitHub branch, commits VFS files, and opens a PR.

Used by the orchestrator agent after all subagents complete their work.
Reads generated files from the DeepAgents filesystem (VFS) and pushes
them to a new branch on the configured GitHub repository.
"""

import os
import uuid
from datetime import datetime
from langchain_core.tools import tool
from github import Github, GithubException

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")  # e.g. "org/repo-name"
BASE_BRANCH = os.environ.get("GITHUB_BASE_BRANCH", "main")


@tool
def submit_pr(
    title: str,
    description: str,
    files: dict[str, str],
) -> str:
    """Create a GitHub branch, commit files, and open a Pull Request.

    Args:
        title: PR title (e.g. "Add S3 → Glue → Redshift data pipeline").
        description: PR body describing what was generated.
        files: Dict mapping file paths to their content.
            Example: {"infra/main.tf": "resource ...", ".github/workflows/deploy.yml": "name: ..."}

    Returns:
        The URL of the created Pull Request, or an error message.
    """
    if not GITHUB_TOKEN:
        return "Error: GITHUB_TOKEN environment variable not set."
    if not GITHUB_REPO:
        return "Error: GITHUB_REPO environment variable not set."
    if not files:
        return "Error: No files provided to commit."

    try:
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        # Create a unique branch name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        branch_name = f"pipeline/{timestamp}-{short_id}"

        # Get the base branch SHA
        base_ref = repo.get_branch(BASE_BRANCH)
        base_sha = base_ref.commit.sha

        # Create the new branch
        repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=base_sha,
        )

        # Commit each file to the branch
        for file_path, content in files.items():
            try:
                # Check if file exists (update it)
                existing = repo.get_contents(file_path, ref=branch_name)
                repo.update_file(
                    path=file_path,
                    message=f"Update {file_path}",
                    content=content,
                    sha=existing.sha,
                    branch=branch_name,
                )
            except GithubException:
                # File doesn't exist yet — create it
                repo.create_file(
                    path=file_path,
                    message=f"Add {file_path}",
                    content=content,
                    branch=branch_name,
                )

        # Open the Pull Request
        pr = repo.create_pull(
            title=title,
            body=description,
            head=branch_name,
            base=BASE_BRANCH,
        )

        return pr.html_url

    except GithubException as e:
        return f"GitHub API error: {e.data.get('message', str(e))}"
    except Exception as e:
        return f"Error creating PR: {str(e)}"
