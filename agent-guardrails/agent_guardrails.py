"""
agent_guardrails.py

A minimal, dependency-free enforcement layer for AI agent tool calls.

Companion code for Article 7: "AI Agents Don't Get Hacked Like Software.
They Get Talked Into It."

The core idea: the model PROPOSES an action. This layer, which sits
entirely outside the model, DECIDES whether that action actually runs.
The agent is never the only thing standing between a request and an
action that can't be undone.

Design principles demonstrated here:
  1. Least-privilege allowlisting (an action must be explicitly known)
  2. High-risk actions require human approval, every time
  3. Every request is logged BEFORE a decision is made, and the outcome
     is logged AFTER — so the audit trail exists even if the action is
     denied, and even if the agent "wanted" something else logged.
  4. The approval/execution functions are injected as callables, so this
     module can be tested without a real human or a real downstream
     system, and can be swapped into any agent framework.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Policy: what the agent is allowed to do, and what needs a human in the loop
# ---------------------------------------------------------------------------

ALLOWED_ACTIONS = {
    "read_file",
    "search_docs",
    "draft_email",       # drafting is safe; SENDING is not (see below)
    "check_order_status",
}

HIGH_RISK_ACTIONS = {
    "send_email",
    "delete_file",
    "transfer_funds",
    "issue_refund",
    "forward_customer_data",
}


# ---------------------------------------------------------------------------
# Audit log: append-only, outside the agent's control
# ---------------------------------------------------------------------------

@dataclass
class AuditLog:
    """An append-only log the agent can write to, but never edit or delete from."""

    entries: list = field(default_factory=list)

    def record(self, event_type: str, **details) -> None:
        self.entries.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "event_type": event_type,
                **details,
            }
        )

    def as_list(self) -> list:
        # Defensive copy — callers get a view, not the real list
        return list(self.entries)


# ---------------------------------------------------------------------------
# The enforcement layer itself
# ---------------------------------------------------------------------------

class ToolCallDenied(Exception):
    """Raised when a proposed action is not permitted to run."""


class AgentGuardrail:
    """
    Wraps agent tool calls with logging, allowlisting, and human approval.

    request_human_approval and run_actual_tool are injected as callables so
    this class can be unit-tested and reused across different agent stacks
    without depending on any particular framework, ticketing system, or
    approval UI.
    """

    def __init__(
        self,
        request_human_approval: Callable[[str, dict], bool],
        run_actual_tool: Callable[[str, dict], dict],
        audit_log: Optional[AuditLog] = None,
        allowed_actions: Optional[set] = None,
        high_risk_actions: Optional[set] = None,
    ):
        self.request_human_approval = request_human_approval
        self.run_actual_tool = run_actual_tool
        self.audit_log = audit_log or AuditLog()
        self.allowed_actions = allowed_actions or ALLOWED_ACTIONS
        self.high_risk_actions = high_risk_actions or HIGH_RISK_ACTIONS

    def execute(self, action_name: str, params: dict, agent_context: dict) -> dict:
        """
        The single entry point every proposed agent action must pass through.

        agent_context should include whatever provenance is available, e.g.
        {"source": "customer_email", "triggered_by": "summarize_inbox_task"}
        so that if this action later needs to be investigated, the log shows
        not just WHAT ran, but WHAT CONTENT caused the agent to propose it.
        """
        # 1. Log the proposal first — before any decision is made.
        self.audit_log.record(
            "action_proposed",
            action=action_name,
            params=params,
            context=agent_context,
        )

        # 2. High-risk actions always require a human, regardless of how
        #    confident the agent is, and regardless of what the triggering
        #    content claimed ("this is pre-approved", "the manager said yes").
        if action_name in self.high_risk_actions:
            approved = self.request_human_approval(action_name, params)
            if not approved:
                self.audit_log.record(
                    "action_denied",
                    action=action_name,
                    reason="human approval withheld",
                )
                raise ToolCallDenied(
                    f"'{action_name}' requires human approval, which was not granted."
                )

        # 3. Anything not explicitly known — allowed or high-risk — is denied
        #    by default. Unknown is not the same as safe.
        elif action_name not in self.allowed_actions:
            self.audit_log.record(
                "action_denied",
                action=action_name,
                reason="action not in allowlist",
            )
            raise ToolCallDenied(f"'{action_name}' is not an allowlisted action.")

        # 4. Execute, and log the outcome too — not just the request.
        result = self.run_actual_tool(action_name, params)
        self.audit_log.record(
            "action_executed",
            action=action_name,
            params=params,
            result=result,
        )
        return result
