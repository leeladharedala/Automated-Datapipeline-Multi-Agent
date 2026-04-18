"""Throttled checkpoint saver — reduces AgentCore Memory API calls.

Wraps AgentCoreMemorySaver with write throttling so that only every Nth
checkpoint write hits the remote API.  Reads always go to the remote saver
so session resumption works correctly.

This dramatically reduces CreateEvent API calls during burst subagent
execution while preserving session resumability.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class ThrottledCheckpointSaver(BaseCheckpointSaver):
    """Checkpoint saver that throttles writes to a remote backend.

    Every ``flush_interval`` writes, the checkpoint is persisted to the
    remote saver.  The final checkpoint of a graph run (containing the
    ``__end__`` channel) is always flushed regardless of the counter.

    All reads are delegated directly to the remote saver so that session
    resume and time-travel work correctly.

    Args:
        remote: The persistent checkpoint saver (e.g. AgentCoreMemorySaver).
        flush_interval: Persist to remote every N writes. Default 5.
    """

    def __init__(
        self,
        remote: BaseCheckpointSaver,
        *,
        flush_interval: int = 5,
    ) -> None:
        super().__init__()
        self._remote = remote
        self._flush_interval = max(1, flush_interval)
        # thread_id -> write count since last flush
        self._write_counts: dict[str, int] = {}
        # track last checkpoint/config per thread for final flush
        self._last_put: dict[str, tuple] = {}

    @property
    def config_specs(self):
        return self._remote.config_specs

    # ── helpers ──────────────────────────────────────────────

    def _thread_id(self, config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "")

    def _is_final(self, new_versions: ChannelVersions) -> bool:
        """Check if this checkpoint contains the __end__ channel (graph done)."""
        return "__end__" in (new_versions or {})

    def _should_flush(self, thread_id: str, new_versions: ChannelVersions) -> bool:
        count = self._write_counts.get(thread_id, 0) + 1
        self._write_counts[thread_id] = count
        if self._is_final(new_versions):
            self._write_counts[thread_id] = 0
            return True
        if count >= self._flush_interval:
            self._write_counts[thread_id] = 0
            return True
        return False

    # ── sync read methods (delegate to remote) ────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self._remote.get_tuple(config)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        yield from self._remote.list(
            config, filter=filter, before=before, limit=limit
        )

    # ── sync write methods (throttled) ───────────────────────

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = self._thread_id(config)
        # Always store the latest for potential final flush
        self._last_put[thread_id] = (config, checkpoint, metadata, new_versions)

        if self._should_flush(thread_id, new_versions):
            logger.debug("Flushing checkpoint to remote for thread %s", thread_id)
            return self._remote.put(config, checkpoint, metadata, new_versions)

        # Return config as-is without persisting
        return config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # Intermediate writes are only needed for durable execution
        # recovery. For subagent bursts we skip them to reduce API calls.
        # The next full checkpoint flush captures the complete state.
        thread_id = self._thread_id(config)
        count = self._write_counts.get(thread_id, 0)
        if count == 0:
            # Only persist writes right after a flush (state is in sync)
            self._remote.put_writes(config, writes, task_id, task_path)

    # ── async read methods (delegate to remote) ──────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await self._remote.aget_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        async for item in self._remote.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    # ── async write methods (throttled) ──────────────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = self._thread_id(config)
        self._last_put[thread_id] = (config, checkpoint, metadata, new_versions)

        if self._should_flush(thread_id, new_versions):
            logger.debug("Async flushing checkpoint to remote for thread %s", thread_id)
            return await self._remote.aput(config, checkpoint, metadata, new_versions)

        return config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = self._thread_id(config)
        count = self._write_counts.get(thread_id, 0)
        if count == 0:
            await self._remote.aput_writes(config, writes, task_id, task_path)
