"""
NetSim Backend Package
======================
A FastAPI-based backend for simulating core networking protocols.

Modules:
    - distance_vector   : Bellman-Ford distance vector routing
    - link_state        : Dijkstra link state routing
    - arp_simulation    : ARP request/reply and cache management
    - dhcp_simulation   : DHCP DORA handshake and lease lifecycle
    - fragmentation_simulation : IP fragmentation and reassembly
    - models            : Shared Pydantic request/response models
    - probability       : Packet loss and link reliability utilities
"""

# --- Routing ---
from .distance_vector import (
    initialize_dv_table,
    process_dv_update,
    process_dv_update_with_logs,
)

from .link_state import (
    generate_lsp,
    calculate_dijkstra,
)

# --- ARP ---
from .arp_simulation import (
    create_arp_node,
    initiate_arp_request,
    send_gratuitous_arp,
    update_arp_cache,
    advance_arp_tick,
    create_broadcast_packets,
)

# --- DHCP ---
from .dhcp_simulation import (
    create_dhcp_node,
    start_dora,
    start_auto_arrivals,
    run_monte_carlo,
    allocate_ip_from_pool,
)

# --- Fragmentation ---
from .fragmentation_simulation import (
    send_packet,
    simulation_tick,
    create_frag_node,
    create_frag_link,
    create_icmp_packet,
)

# --- Models ---
from .models import (
    RouterNode,
    RouterLink,
    DVEntry,
    DVPayloadEntry,
    LSP,
    LSPLink,
    RouteMessage,
    InitializeRequest,
    InitializeResponse,
    StepRequest,
    StepResponse,
    ProbabilityRequest,
    ProbabilityResponse,
)

__all__ = [
    # Routing
    "initialize_dv_table",
    "process_dv_update",
    "process_dv_update_with_logs",
    "generate_lsp",
    "calculate_dijkstra",
    # ARP
    "create_arp_node",
    "initiate_arp_request",
    "send_gratuitous_arp",
    "update_arp_cache",
    "advance_arp_tick",
    "create_broadcast_packets",
    # DHCP
    "create_dhcp_node",
    "start_dora",
    "start_auto_arrivals",
    "run_monte_carlo",
    "allocate_ip_from_pool",
    # Fragmentation
    "send_packet",
    "simulation_tick",
    "create_frag_node",
    "create_frag_link",
    "create_icmp_packet",
    # Models
    "RouterNode",
    "RouterLink",
    "DVEntry",
    "DVPayloadEntry",
    "LSP",
    "LSPLink",
    "RouteMessage",
    "InitializeRequest",
    "InitializeResponse",
    "StepRequest",
    "StepResponse",
    "ProbabilityRequest",
    "ProbabilityResponse",
]
