"""How each agent run ended, read from the parent's records and journals.

An agent's own log does not say how it ended — the parent records the
outcome, either as the result of the tool call that started the run, as a
background task notification, or in a workflow journal. Nested agents
report to their parent agent's file, so every file is read for events, all
keyed by agent id. Where only a tool-call id is known, the sidecars'
`toolUseId` maps it to the agent.

When no record exists, the run's own file can still suggest an ending (a
final `end_turn` turn, or an API error); that inference is marked as such
and never presented as a recorded fact.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, JsonValue

from agent_scorecard.logfile import DecodedLine, LineKind
from agent_scorecard.models import (
    Lifecycle,
    LifecycleSource,
    RawUserLine,
    as_dict,
    as_str,
    text_blocks,
    tool_result_blocks,
)
from agent_scorecard.runs import AgentRun, RunFacts

logger = logging.getLogger(__name__)


class LifecycleEvent(BaseModel):
    """One recorded final event for one agent run. `timestamp` is None for
    journal events, which rank below any timestamped event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    kind: Lifecycle
    source: LifecycleSource
    timestamp: datetime.datetime | None = None
    reason: str | None = None
    result_text: str | None = None


_REASON_CAP = 120

# The notification is a small tagged text block; these pull out the fields
# the lifecycle needs. DOTALL because <result> can span lines.
_TASK_ID_RE = re.compile(r"<task-id>(.*?)</task-id>", re.DOTALL)
_STATUS_RE = re.compile(r"<status>(.*?)</status>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)

# Reasons must never contain the task description; the summary always
# leads with the description, so when the sidecar's description is unknown,
# everything up to the last quote before the outcome marker goes.
_REASON_MARKERS = ("failed:", "finished", "was stopped by user")


def first_reason_line(text: str, cap: int = _REASON_CAP) -> str:
    """The first non-blank line of an error text, cut to the cap."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:cap]
    return text.strip()[:cap]


def reason_from_summary(summary: str, description: str | None) -> str:
    """Strip the `Agent "<description>"` prefix from a notification summary,
    leaving at most the cap's worth of reason."""
    text = summary.strip()
    if description is not None:
        prefix = f'Agent "{description}"'
        if text.startswith(prefix):
            return text[len(prefix) :].strip()[:_REASON_CAP]
    cut = len(text)
    for marker in _REASON_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    quote = text[:cut].rfind('" ')
    if quote != -1:
        text = text[quote + 1 :]
    return text.strip()[:_REASON_CAP]


def _notification_kind(status: str) -> Lifecycle | None:
    return {
        "completed": Lifecycle.COMPLETED,
        "failed": Lifecycle.FAILED,
        "killed": Lifecycle.KILLED,
        "stopped": Lifecycle.STOPPED,
    }.get(status)


class LifecycleCollector:
    """Collects recorded final events from every file the scan reads."""

    def __init__(
        self,
        *,
        tool_id_to_agent: dict[str, str],
        descriptions: dict[str, str],
    ) -> None:
        self._tool_id_to_agent = tool_id_to_agent
        self._descriptions = descriptions
        self._events: list[LifecycleEvent] = []
        self._unexpected_statuses_logged: set[str] = set()

    def feed(self, line: DecodedLine) -> None:
        if line.kind is not LineKind.NON_ASSISTANT or line.user is None:
            return
        user = line.user
        tool_use_result = as_dict(user.tool_use_result)
        if tool_use_result is not None:
            self._feed_foreground(user, tool_use_result)
            return
        if user.origin is not None and user.origin.kind == "task-notification":
            self._feed_notification(user)

    def feed_journal(self, raw: dict[str, JsonValue]) -> None:
        """One `journal.jsonl` entry: `result` and `failed` are outcomes;
        `started` only says the run began."""
        agent_id = as_str(raw.get("agentId"))
        kind = as_str(raw.get("type"))
        if agent_id is None:
            return
        if kind == "result":
            result = raw.get("result")
            self._events.append(
                LifecycleEvent(
                    agent_id=agent_id,
                    kind=Lifecycle.COMPLETED,
                    source=LifecycleSource.WORKFLOW_JOURNAL,
                    result_text=result if isinstance(result, str) else None,
                )
            )
        elif kind == "failed":
            self._events.append(
                LifecycleEvent(
                    agent_id=agent_id,
                    kind=Lifecycle.FAILED,
                    source=LifecycleSource.WORKFLOW_JOURNAL,
                )
            )

    def finish(self) -> tuple[LifecycleEvent, ...]:
        return tuple(self._events)

    def _feed_foreground(self, user: RawUserLine, result: dict[str, JsonValue]) -> None:
        """One `toolUseResult` object on the parent's tool result line.

        `async_launched` is a launch, not an outcome — the notification
        that follows carries the outcome. An errored result, or any status
        other than `completed`, is a failure.
        """
        status = as_str(result.get("status"))
        if status == "async_launched":
            return
        agent_id = as_str(result.get("agentId")) or None
        if agent_id is None:
            blocks = tool_result_blocks(user.message.content)
            tool_use_id = blocks[0].tool_use_id if blocks else None
            agent_id = self._tool_id_to_agent.get(tool_use_id) if tool_use_id else None
        if agent_id is None:
            return
        blocks = tool_result_blocks(user.message.content)
        errored = next((block for block in blocks if block.is_error), None)
        if errored is None and status == "completed":
            self._events.append(
                LifecycleEvent(
                    agent_id=agent_id,
                    kind=Lifecycle.COMPLETED,
                    source=LifecycleSource.FOREGROUND,
                    timestamp=user.timestamp,
                    result_text="".join(text_blocks(result.get("content"))) or None,
                )
            )
            return
        if status is not None:
            self._log_unexpected_status(status)
        reason_source = (
            first_reason_line("".join(text_blocks(errored.content)))
            if errored is not None
            else first_reason_line("".join(text_blocks(result.get("content"))))
        )
        self._events.append(
            LifecycleEvent(
                agent_id=agent_id,
                kind=Lifecycle.FAILED,
                source=LifecycleSource.FOREGROUND,
                timestamp=user.timestamp,
                reason=reason_source or None,
            )
        )

    def _feed_notification(self, user: RawUserLine) -> None:
        """A background agent's outcome. Only `user` lines with
        `origin.kind == "task-notification"` are read: the same text is
        copied into `queue-operation` and `attachment` lines, and reading
        those would count every outcome twice. Notifications about things
        other than agent runs carry task ids matching no run and are
        dropped when events join runs."""
        content = user.message.content
        if not isinstance(content, str):
            return
        task_id = _TASK_ID_RE.search(content)
        status = _STATUS_RE.search(content)
        if task_id is None or status is None:
            return
        agent_id = task_id.group(1).strip()
        status_text = status.group(1).strip()
        kind = _notification_kind(status_text)
        if kind is None:
            self._log_unexpected_status(status_text)
            kind = Lifecycle.FAILED
        summary_match = _SUMMARY_RE.search(content)
        summary = summary_match.group(1).strip() if summary_match else ""
        result_match = _RESULT_RE.search(content)
        result_text = result_match.group(1).strip() if result_match else ""
        self._events.append(
            LifecycleEvent(
                agent_id=agent_id,
                kind=kind,
                source=LifecycleSource.NOTIFICATION,
                timestamp=user.timestamp,
                reason=reason_from_summary(summary, self._descriptions.get(agent_id)),
                result_text=result_text or None,
            )
        )

    def _log_unexpected_status(self, status: str) -> None:
        if status in self._unexpected_statuses_logged:
            return
        self._unexpected_statuses_logged.add(status)
        logger.warning("Unexpected agent outcome status %r; treated as failed.", status)


class ResolvedLifecycle(BaseModel):
    """The one lifecycle answer for a run, plus where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lifecycle: Lifecycle = Lifecycle.UNKNOWN
    source: LifecycleSource = LifecycleSource.NONE
    reason: str | None = None


def resolve_lifecycle(facts: RunFacts, events: Sequence[LifecycleEvent]) -> ResolvedLifecycle:
    """The latest recorded final event wins; journal lines carry no
    timestamp, so they rank below any timestamped event. With no record at
    all, the sidecar's `stoppedByUser` decides, then the run's own file is
    consulted — marked as inferred — and only then is the lifecycle
    unknown."""
    if events:
        ordered = sorted(events, key=_event_sort_key)
        # Ascending by (timestamped, time): the last entry is the latest
        # timestamped event, and any journal line ranks below all of them.
        winner = ordered[-1]
        return ResolvedLifecycle(lifecycle=winner.kind, source=winner.source, reason=winner.reason)
    if facts.stopped_by_user:
        return ResolvedLifecycle(
            lifecycle=Lifecycle.KILLED,
            source=LifecycleSource.SIDECAR,
            reason="stopped by user",
        )
    if facts.last_assistant_is_api_error:
        return ResolvedLifecycle(
            lifecycle=Lifecycle.FAILED,
            source=LifecycleSource.INFERRED,
            reason="API error",
        )
    if facts.last_stop_reason == "end_turn" and not facts.last_stop_is_api_error:
        return ResolvedLifecycle(lifecycle=Lifecycle.COMPLETED, source=LifecycleSource.INFERRED)
    return ResolvedLifecycle()


def _event_sort_key(event: LifecycleEvent) -> tuple[bool, datetime.datetime]:
    # Journal events (timestamp None) sort before any timestamped event, so
    # the last entry after this ascending sort is the latest timestamp.
    return (
        event.timestamp is not None,
        event.timestamp or datetime.datetime.min.replace(tzinfo=datetime.UTC),
    )


def attach_lifecycles(
    runs: Sequence[AgentRun],
    facts: Sequence[RunFacts],
    events: Sequence[LifecycleEvent],
) -> tuple[AgentRun, ...]:
    """Resolve one lifecycle per run and return the runs with it filled in.

    Events keyed to agent ids matching no run — background commands, other
    sessions' agents — are dropped here: a notification about something
    that never ran as an agent says nothing this tool should report.
    """
    facts_by_id = {item.agent_id: item for item in facts}
    events_by_id: dict[str, list[LifecycleEvent]] = {}
    for event in events:
        if event.agent_id in facts_by_id:
            events_by_id.setdefault(event.agent_id, []).append(event)
    resolved: list[AgentRun] = []
    for run in runs:
        item = facts_by_id.get(run.agent_id)
        answer = (
            resolve_lifecycle(item, events_by_id.get(run.agent_id, ()))
            if item is not None
            else ResolvedLifecycle()
        )
        resolved.append(
            run.model_copy(
                update={
                    "lifecycle": answer.lifecycle,
                    "lifecycle_source": answer.source,
                    "reason": answer.reason,
                }
            )
        )
    return tuple(resolved)
