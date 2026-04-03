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
    "AWS_DOCS_MCP_COMMAND", "uvx"
)
AWS_DOCS_MCP_ARGS = os.environ.get(
    "AWS_DOCS_MCP_ARGS",
    "awslabs.aws-documentation-mcp-server@latest",
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


async def load_gateway_tools() -> list:
    """Load all tools from both MCP servers.

    Returns a list of LangChain-compatible tools that can be
    passed to create_deep_agent's tools parameter or bound
    to a subagent config.
    """
    client = create_mcp_client()
    tools = await client.get_tools()
    return tools
