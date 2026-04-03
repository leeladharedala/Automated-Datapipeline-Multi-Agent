"""
AgentCore Memory setup — creates the memory resource with extraction strategies.

Run this ONCE during initial deployment to configure the memory resource
with three strategies:
1. UserPreference — extracts user preferences (region, framework, naming conventions)
2. Semantic — extracts facts and knowledge (pipeline architectures, schema details)
3. Summary — creates running session summaries for long conversations

After running this, set the returned Memory ID as AGENTCORE_MEMORY_ID env var.
"""

import os
import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
MEMORY_NAME = os.environ.get("AGENTCORE_MEMORY_NAME", "DataPipelineAgentMemory")


def create_memory_resource() -> str:
    """Create an AgentCore Memory resource with all three strategies.

    Returns the Memory ID to be used as AGENTCORE_MEMORY_ID.
    """
    client = boto3.client(
        "bedrock-agentcore-control",
        region_name=REGION,
    )

    response = client.create_memory(
        name=MEMORY_NAME,
        memoryStrategies=[
            # 1. User preferences — architectural choices, cloud region,
            #    framework preferences, naming conventions
            {
                "userPreferenceMemoryStrategy": {
                    "name": "UserPreferenceExtractor",
                    "namespaceTemplates": [
                        "/users/{actorId}/preferences/",
                    ],
                }
            },
            # 2. Semantic facts — pipeline architectures, data schemas,
            #    resource names, past decisions
            {
                "semanticMemoryStrategy": {
                    "name": "ArchitecturalFactExtractor",
                    "namespaceTemplates": [
                        "/users/{actorId}/facts/",
                    ],
                }
            },
            # 3. Session summaries — condensed recaps of long
            #    multi-agent pipeline generation sessions
            {
                "summaryMemoryStrategy": {
                    "name": "SessionSummarizer",
                    "namespaceTemplates": [
                        "/summaries/{actorId}/{sessionId}/",
                    ],
                }
            },
        ],
    )

    memory_id = response["memoryId"]
    print(f"Memory resource created: {memory_id}")
    print(f"Set this as your env var: export AGENTCORE_MEMORY_ID={memory_id}")
    return memory_id


if __name__ == "__main__":
    create_memory_resource()
