"""
Gateway MCP Adapter — connects to Terraform Registry MCP and AWS Docs MCP.

Uses langchain-mcp-adapters to load tools from both MCP servers over stdio
transport. Each server is loaded independently so that a failure in one does
not affect the other.

Terraform MCP is essential — if it fails, the error is raised.
AWS Docs MCP is non-essential — if it fails, a warning is logged and the
agent continues with Terraform tools only (graceful degradation).
"""

import logging
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# MCP server commands — configurable via env vars for different environments
TERRAFORM_MCP_COMMAND = os.environ.get(
    "TERRAFORM_MCP_COMMAND", "npx"
)
TERRAFORM_MCP_ARGS = os.environ.get(
    "TERRAFORM_MCP_ARGS", "terraform-mcp-server"
).split()

AWS_DOCS_MCP_COMMAND = os.environ.get(
    "AWS_DOCS_MCP_COMMAND", "uvx"
)
AWS_DOCS_MCP_ARGS = os.environ.get(
    "AWS_DOCS_MCP_ARGS",
    "awslabs.aws-documentation-mcp-server@latest",
).split()

# AWS Docs MCP environment — per https://github.com/awslabs/mcp
AWS_DOCS_PARTITION = os.environ.get("AWS_DOCUMENTATION_PARTITION", "aws")
AWS_DOCS_USER_AGENT = os.environ.get(
    "MCP_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)


def create_mcp_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient for Terraform + AWS Docs MCP servers.

    .. deprecated::
        This function is kept for backward compatibility. New code should
        use ``load_gateway_tools()`` which loads servers independently.
    """
    return MultiServerMCPClient(
        {
            "terraform": {
                "command": TERRAFORM_MCP_COMMAND,
                "args": TERRAFORM_MCP_ARGS,
                "transport": "stdio",
                "env": {
                    "FASTMCP_LOG_LEVEL": "ERROR",
                },
            },
            "aws_docs": {
                "command": AWS_DOCS_MCP_COMMAND,
                "args": AWS_DOCS_MCP_ARGS,
                "transport": "stdio",
                "env": {
                    "FASTMCP_LOG_LEVEL": "ERROR",
                    "AWS_DOCUMENTATION_PARTITION": AWS_DOCS_PARTITION,
                    "MCP_USER_AGENT": AWS_DOCS_USER_AGENT,
                },
            },
        }
    )


async def load_gateway_tools() -> tuple[list, list]:
    """Load tools from MCP servers independently.

    Each server is loaded in its own try/except block for full isolation.
    Terraform MCP is essential — failure raises. AWS Docs MCP is
    non-essential — failure logs a warning and continues.

    Returns:
        A tuple of (clients, tools) where clients is a list of
        successfully connected MCP client instances (caller must keep
        them alive) and tools is the merged tool list.
    """
    clients: list = []
    all_tools: list = []

    # Terraform MCP (essential) — must succeed
    terraform_client = MultiServerMCPClient({
        "terraform": {
            "command": TERRAFORM_MCP_COMMAND,
            "args": TERRAFORM_MCP_ARGS,
            "transport": "stdio",
            "env": {
                "FASTMCP_LOG_LEVEL": "ERROR",
            },
        },
    })
    try:
        terraform_tools = await terraform_client.get_tools()
        clients.append(terraform_client)
        all_tools.extend(terraform_tools)
    except Exception as exc:
        logger.error("Terraform MCP server failed to start: %s", exc)
        raise

    # AWS Docs MCP (non-essential) — graceful degradation
    aws_docs_client = MultiServerMCPClient({
        "aws_docs": {
            "command": AWS_DOCS_MCP_COMMAND,
            "args": AWS_DOCS_MCP_ARGS,
            "transport": "stdio",
            "env": {
                "FASTMCP_LOG_LEVEL": "ERROR",
                "AWS_DOCUMENTATION_PARTITION": AWS_DOCS_PARTITION,
                "MCP_USER_AGENT": AWS_DOCS_USER_AGENT,
            },
        },
    })
    try:
        aws_docs_tools = await aws_docs_client.get_tools()
        clients.append(aws_docs_client)
        all_tools.extend(aws_docs_tools)
    except Exception as exc:
        logger.warning("AWS Docs MCP server failed to start: %s", exc)

    return clients, all_tools
