import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# List of active log subscribers
_subscribers: list[Callable[[str, str], None]] = []


def subscribe_realtime_logs(callback: Callable[[str, str], None]) -> None:
    """Register a callback to receive realtime logs."""
    if callback not in _subscribers:
        _subscribers.append(callback)


def unsubscribe_realtime_logs(callback: Callable[[str, str], None]) -> None:
    """Unregister a log subscriber."""
    if callback in _subscribers:
        _subscribers.remove(callback)


def log_subagent_progress(agent_name: str, message: str) -> None:
    """Log progress for a sub-agent and dispatch to all active subscribers.

    Example:
        log_subagent_progress("iac-agent", "[validate] Running terraform plan...")
    """
    logger.info("[%s] %s", agent_name, message)
    for sub in _subscribers:
        try:
            sub(agent_name, message)
        except Exception as exc:
            logger.warning("Failed to dispatch log to subscriber: %s", exc)

    # Also publish into the LangGraph "custom" stream so the ingress relay
    # (stream_run_relay joining this run on the co-located server) can forward
    # the line to the dashboard as realtime_logs. No-op outside a run context.
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer is not None:
            writer({"agent": agent_name, "message": message})
    except Exception:
        pass
