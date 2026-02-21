"""
Pydantic models mirroring the TypeScript interfaces in src/simulation/Models.ts
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel


# ── Shared ──────────────────────────────────────────────────────────────────

class Position(BaseModel):
    x: float
    y: float


class RouterNode(BaseModel):
    id: str
    name: str
    position: Position


class RouterLink(BaseModel):
    id: str
    sourceId: str
    targetId: str
    cost: int
    status: Optional[Literal['up', 'down']] = 'up'
    downTicks: Optional[int] = None


# ── Distance Vector ──────────────────────────────────────────────────────────

class DVEntry(BaseModel):
    destinationId: str
    nextHopId: Optional[str]
    cost: int


class DVPayloadEntry(BaseModel):
    destinationId: str
    cost: int


# RoutingTableDV: Dict[destinationId -> DVEntry]
# DVPayload:       Dict[destinationId -> DVPayloadEntry]


# ── Link State ───────────────────────────────────────────────────────────────

class LSPLink(BaseModel):
    targetId: str
    cost: int


class LSP(BaseModel):
    sourceId: str
    sequenceNumber: int
    links: List[LSPLink]


# ── Route Messages ───────────────────────────────────────────────────────────

class RouteMessage(BaseModel):
    id: str
    sourceNodeId: str
    targetNodeId: str
    linkId: str
    type: Literal['LSP_FLOOD', 'DV_UPDATE']
    payload: Any
    status: Optional[Literal['delivered', 'dropped']] = 'delivered'


# ── API Request / Response schemas ───────────────────────────────────────────

ProtocolType = Literal['DV', 'LS']
DistModeType = Literal['Uniform', 'Normal', 'Binomial', 'Poisson']


class InitializeRequest(BaseModel):
    nodes: List[RouterNode]
    links: List[RouterLink]
    protocol: ProtocolType


class InitializeResponse(BaseModel):
    dvTables: Dict[str, Dict[str, DVEntry]]       # nodeId -> { destId -> DVEntry }
    lsdb: Dict[str, Dict[str, LSP]]               # nodeId -> { sourceId -> LSP }
    lsFloodQueues: Dict[str, List[LSP]]           # nodeId -> [LSP]
    lspSeqCounter: int


class StepRequest(BaseModel):
    nodes: List[RouterNode]
    links: List[RouterLink]
    dvTables: Dict[str, Dict[str, Dict]]          # raw dicts so we can reconstruct
    lsdb: Dict[str, Dict[str, Dict]]
    lsFloodQueues: Dict[str, List[Dict]]
    protocol: ProtocolType
    packetLossRate: float = 0.0
    packetDist: DistModeType = 'Uniform'
    linkFailureRate: float = 0.0
    linkFailDist: DistModeType = 'Uniform'
    linkRecoverDist: DistModeType = 'Uniform'
    lspSeqCounter: int = 1
    iterationCount: int = 0


class CalculationLog(BaseModel):
    id: str
    message: str


class ActiveCalculation(BaseModel):
    nodeId: str
    equation: str


class StepResponse(BaseModel):
    # Updated topology (links may change due to weather)
    links: List[RouterLink]

    # Updated tables
    dvTables: Dict[str, Dict[str, DVEntry]]
    lsdb: Dict[str, Dict[str, LSP]]
    lsFloodQueues: Dict[str, List[LSP]]

    # Messages generated this tick (for animation)
    activeMessages: List[RouteMessage]

    # Log / UI overlays
    calculationLogs: List[CalculationLog]
    activeCalculations: List[ActiveCalculation]

    # State
    simulationState: Literal['READY', 'COMPUTING', 'CONVERGED']
    iterationCount: int
    lspSeqCounter: int


class ProbabilityRequest(BaseModel):
    mode: DistModeType
    p: float


class ProbabilityResponse(BaseModel):
    result: bool
