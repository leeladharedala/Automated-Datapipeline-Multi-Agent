"""SanitizedChatAnthropic — ChatAnthropic that survives empty thinking blocks.

claude-sonnet-5 (adaptive thinking) sometimes emits an assistant message whose
thinking/reasoning content block carries a signature but NO text. Replaying
such a message on the next tool-loop turn is rejected by the Anthropic API
with 400 "messages.N.content.0.thinking.thinking: Field required" — observed
repeatedly killing sub-agent runs (and latent for the Supervisor, which
replays its checkpointed history every turn).

Instead of guessing which layer drops the text (streaming aggregation,
checkpoint serde, server round-trip — the failure reproduced with several of
them ruled out), strip the invalid blocks at the model boundary: every
generation path funnels through `_generate/_agenerate/_stream/_astream`, so
sanitizing there is complete. Valid (non-empty) thinking blocks pass through
untouched, preserving prompt-cache and interleaved-thinking behavior.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage

_THINKING_TYPES = ("thinking", "reasoning")


def strip_empty_thinking_blocks(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop thinking/reasoning content blocks that have no text.

    An empty thinking block is invalid on replay regardless of provenance;
    omitting a thinking block entirely is accepted by the API, so dropping is
    strictly safer than re-serializing.
    """
    out: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            blocks: list[Any] = []
            for block in message.content:
                if isinstance(block, dict) and block.get("type") in _THINKING_TYPES:
                    text = block.get("thinking") or block.get("reasoning") or ""
                    if not text:
                        continue
                blocks.append(block)
            if len(blocks) != len(message.content):
                message = message.model_copy(update={"content": blocks})
        out.append(message)
    return out


class SanitizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic that sanitizes replayed history before every call."""

    def _generate(self, messages, *args, **kwargs):
        return super()._generate(strip_empty_thinking_blocks(messages), *args, **kwargs)

    async def _agenerate(self, messages, *args, **kwargs):
        return await super()._agenerate(
            strip_empty_thinking_blocks(messages), *args, **kwargs
        )

    def _stream(self, messages, *args, **kwargs):
        return super()._stream(strip_empty_thinking_blocks(messages), *args, **kwargs)

    def _astream(self, messages, *args, **kwargs):
        return super()._astream(strip_empty_thinking_blocks(messages), *args, **kwargs)
