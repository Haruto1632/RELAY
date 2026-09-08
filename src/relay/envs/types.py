"""Core state records for the RELAY environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np

AGENT_SLOTS = ("scout_0", "ambulance_0", "ambulance_1", "fireman_0", "fireman_1")


class Role(StrEnum):
    SCOUT = "scout"
    AMBULANCE = "ambulance"
    FIREMAN = "fireman"


class IncidentKind(StrEnum):
    VICTIM = "victim"
    FIRE = "fire"


class IncidentStatus(StrEnum):
    ACTIVE = "active"
    RESERVED = "reserved"
    RESOLVED = "resolved"


class PhysicalAction(IntEnum):
    STAY = 0
    MOVE_N = 1
    MOVE_E = 2
    MOVE_S = 3
    MOVE_W = 4
    INTERACT = 5


def role_for(agent_id: str) -> Role:
    if agent_id.startswith("scout"):
        return Role.SCOUT
    if agent_id.startswith("ambulance"):
        return Role.AMBULANCE
    if agent_id.startswith("fireman"):
        return Role.FIREMAN
    raise ValueError(f"unsupported agent ID {agent_id!r}")


@dataclass(slots=True)
class AgentState:
    agent_id: str
    role: Role
    x: int
    y: int
    busy_incident_id: str | None = None
    busy_remaining: int = 0
    last_move_blocked: bool = False
    last_interaction_started: bool = False
    last_physical_action: int = 0


@dataclass(slots=True)
class IncidentState:
    incident_id: str
    kind: IncidentKind
    x: int
    y: int
    status: IncidentStatus = IncidentStatus.ACTIVE
    reserved_by: str | None = None


@dataclass(slots=True)
class Packet:
    send_tick: int
    delivery_tick: int
    sender_id: str
    sender_role: Role
    recipient_id: str
    payload: np.ndarray
    attempt_cost: float
    drop_sample: float
    dropped: bool


@dataclass(slots=True)
class WorldState:
    seed: int
    scenario_id: str
    tick: int
    walls: np.ndarray
    staging: np.ndarray
    agents: dict[str, AgentState]
    incidents: dict[str, IncidentState]
    packet_queue: list[Packet]
    delivered_packets: dict[str, list[Packet]]
    channel_rng_state: dict[str, Any]
    tie_break_order: tuple[str, ...]
    discovered_incidents: dict[str, str]


class ScenarioGenerationError(RuntimeError):
    """Raised when no valid scenario can be generated within the fixed cap."""


class EnvironmentStateError(RuntimeError):
    """Raised when step/reset lifecycle rules are violated."""
