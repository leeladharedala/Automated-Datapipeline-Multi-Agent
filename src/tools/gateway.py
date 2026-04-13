"""
Gateway MCP Adapter — connects to Terraform Registry MCP and AWS Docs MCP.

Uses langchain-mcp-adapters to load tools from both MCP servers over stdio
transport. These tools are bound to the IaC subagent so it can look up
resource schemas and AWS documentation before generating Terraform code.
"""

import os
from langchain_mcp_adapters.client import MultiServerMCPClient

# MCP server commands — configurable via env vars for different environments
TERRAFORM_MCP_COMMAND = os.environ.get(
    "TERRAFORM_MCP_COMMAND", "npx"
)
TERRAFORM_MCP_ARGS = os.environ.get(
    "TERRAFORM_MCP_ARGS", "@hashicorp/terraform-mcp-server"
).split()

AWS_DOCS_MCP_COMMAND = os.environ.get(
    "AWS_DOCS_MCP_COMMAND", "python"
)
AWS_DOCS_MCP_ARGS = os.environ.get(
    "AWS_DOCS_MCP_ARGS",
    "-m awslabs.aws_documentation_mcp_server",
).split()


def create_mcp_client() -> MultiServerMCPClient:
    """Create a MultiServerMCPClient for Terraform + AWS Docs MCP servers."""
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
                },
            },
        }
    )


async def load_gateway_tools() -> tuple["MultiServerMCPClient", list]:
    """Load all tools from both MCP servers.

    Returns a (client, tools) tuple. The caller MUST keep the client
    reference alive for the lifetime of tool usage — the MCP stdio
    processes are tied to the client's async context.

    Usage::

        client, tools = await load_gateway_tools()
        # ... use tools while client is alive ...
        # client will be cleaned up when no longer referenced

    Returns:
        A tuple of (MultiServerMCPClient, list of LangChain-compatible tools).
    """
    client = create_mcp_client()
    await client.__aenter__()
    tools = client.get_tools()
    return client, tools
