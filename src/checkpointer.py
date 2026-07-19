"""FastAgentCoreMemorySaver — AgentCoreMemorySaver with O(N) delta-history reads.

deepagents 0.6 stores the Supervisor's ``messages`` state in langgraph's beta
``DeltaChannel``: checkpoint blobs hold only an ``_empty`` sentinel and the
real history is reconstructed by replaying every ancestor checkpoint's writes
via ``BaseCheckpointSaver.get_delta_channel_history``.

The base-class default walks the parent chain calling ``get_tuple`` once per
ancestor — and ``AgentCoreMemorySaver.get_tuple`` re-fetches the session's
ENTIRE event list from AgentCore Memory on every call. That makes every state
load O(N²) in API traffic: measured ~10 minutes for a thread with ~250
checkpoints, which stalled every Supervisor turn, status poll, and
``aget_state`` call in production.

This subclass fetches the event history ONCE, then walks the parent chain in
memory, producing byte-identical results to the default implementation
(same write ordering, same seed semantics). Any internal-API mismatch falls
back to the slow-but-correct base implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph_checkpoint_aws import AgentCoreMemorySaver

logger = logging.getLogger(__name__)


class FastAgentCoreMemorySaver(AgentCoreMemorySaver):
    """AgentCoreMemorySaver with a single-fetch ``get_delta_channel_history``."""

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        if not channels:
            return {}
        try:
            return self._fast_delta_history(config=config, channels=channels)
        except Exception as exc:  # pragma: no cover - depends on saver internals
            logger.warning(
                "Fast delta-history walk failed (%s); falling back to the "
                "per-ancestor base implementation",
                exc,
            )
            return super().get_delta_channel_history(config=config, channels=channels)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        # The saver's other async methods also run their sync counterparts in
        # an executor; the base async default would instead await aget_tuple
        # per ancestor (one full event fetch each), which is what we're fixing.
        from langchain_core.runnables.config import run_in_executor

        return await run_in_executor(
            None,
            partial(self.get_delta_channel_history, config=config, channels=channels),
        )

    def _fast_delta_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        """One event fetch + in-memory parent-chain walk.

        Mirrors ``BaseCheckpointSaver.get_delta_channel_history`` exactly:
        start at the target checkpoint's PARENT (the target's own pending
        writes are the in-flight step and are excluded), collect each
        ancestor's writes for the requested channels (reversed per ancestor,
        reversed again at the end → chronological), and stop collecting a
        channel at the nearest ancestor whose ``channel_values`` contain it
        (that value becomes the seed).
        """
        from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
            CheckpointerConfig,
        )

        checkpoint_config = CheckpointerConfig.from_runnable_config(dict(config))

        events = self.checkpoint_event_client.get_events(
            checkpoint_config.session_id,
            checkpoint_config.actor_id,
            self.limit,
            self.max_results,
        )
        checkpoints, writes_by_checkpoint, channel_data = self.processor.process_events(
            events
        )

        empty: dict[str, Any] = {ch: {"writes": []} for ch in channels}
        if not checkpoints:
            return empty

        if checkpoint_config.checkpoint_id:
            target = checkpoints.get(checkpoint_config.checkpoint_id)
            if target is None:
                return empty
        else:
            target = checkpoints[max(checkpoints.keys())]

        collected: dict[str, list[Any]] = {ch: [] for ch in channels}
        seeds: dict[str, Any] = {}
        remaining = set(channels)
        seen: set[str] = set()  # cycle guard
        cursor_id = target.parent_checkpoint_id

        while cursor_id and remaining and cursor_id not in seen:
            seen.add(cursor_id)
            event = checkpoints.get(cursor_id)
            if event is None:
                break
            tup = self.processor.build_checkpoint_tuple(
                event,
                writes_by_checkpoint.get(cursor_id, []),
                channel_data,
                checkpoint_config,
            )
            if tup.pending_writes:
                for write in reversed(tup.pending_writes):
                    ch = write[1]
                    if ch in remaining:
                        collected[ch].append(write)
            for ch in list(remaining):
                if ch in tup.checkpoint["channel_values"]:
                    seeds[ch] = tup.checkpoint["channel_values"][ch]
                    remaining.discard(ch)
            cursor_id = event.parent_checkpoint_id

        result: dict[str, Any] = {}
        for ch in channels:
            entry: dict[str, Any] = {"writes": list(reversed(collected[ch]))}
            if ch in seeds:
                entry["seed"] = seeds[ch]
            result[ch] = entry
        return result
