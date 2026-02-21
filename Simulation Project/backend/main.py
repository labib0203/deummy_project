"""
FastAPI Backend — Network Protocol Simulation Engine
Exposes REST endpoints that mirror the logic in the Zustand stores.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import simulation modules ──────────────────────────────────────────────
from models import (
    InitializeRequest, InitializeResponse,
    StepRequest, StepResponse,
    ProbabilityRequest, ProbabilityResponse,
    DVEntry, LSP,
    CalculationLog, ActiveCalculation, RouteMessage,
)
from probability import check_probability, generate_realistic_edge_weight
from distance_vector import INF, initialize_dv_table, process_dv_update_with_logs
from link_state import generate_lsp, calculate_dijkstra

# ── ARP / DHCP / Fragmentation ────────────────────────────────────────────
from arp_simulation import (
    create_arp_node, create_broadcast_packets, initiate_arp_request,
    send_gratuitous_arp, advance_arp_tick,
)
from dhcp_simulation import (
    create_dhcp_node, start_dora, start_auto_arrivals, run_monte_carlo,
)
from fragmentation_simulation import (
    create_frag_node, create_frag_link, send_packet as frag_send_packet,
    simulation_tick as frag_tick, STARTING_IDENTIFICATION,
)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Network Protocol Simulation API",
    description="Python backend that powers the routing, ARP, DHCP, and fragmentation simulators.",
    version="1.0.0",
)

# Allow all origins for local development (Vite runs on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Network Simulation API is running"}


# ─────────────────────────────────────────────────────────────────────────────
# Routing Protocol Endpoints  (Distance Vector + Link State)
# ─────────────────────────────────────────────────────────────────────────────

lsp_seq_counter = {"value": 1}  # Module-level mutable counter


@app.post("/api/routing/initialize", response_model=InitializeResponse)
def routing_initialize(req: InitializeRequest):
    """
    Initialize DV or LS routing tables for a given topology.
    Mirrors initializeDVTables() in useNetworkStore.ts.
    """
    nodes = [n.model_dump() for n in req.nodes]
    links = [l.model_dump() for l in req.links]
    protocol = req.protocol

    dv_tables: Dict[str, Dict[str, dict]] = {}
    lsdb: Dict[str, Dict[str, dict]] = {}
    ls_flood_queues: Dict[str, List[dict]] = {}

    for node in nodes:
        node_id = node["id"]
        if protocol == "DV":
            dv_tables[node_id] = initialize_dv_table(node_id, links)
        elif protocol == "LS":
            lsp_seq_counter["value"] += 1
            lsp = generate_lsp(node_id, lsp_seq_counter["value"], links)
            lsdb[node_id] = {node_id: lsp}
            ls_flood_queues[node_id] = [lsp]
            dv_tables[node_id] = calculate_dijkstra(node_id, lsdb[node_id])

    # Convert raw dicts to DVEntry objects
    dv_tables_out: Dict[str, Dict[str, DVEntry]] = {
        node_id: {dest_id: DVEntry(**entry) for dest_id, entry in table.items()}
        for node_id, table in dv_tables.items()
    }
    lsdb_out: Dict[str, Dict[str, LSP]] = {
        node_id: {src_id: LSP(**lsp) for src_id, lsp in db.items()}
        for node_id, db in lsdb.items()
    }
    ls_flood_queues_out: Dict[str, List[LSP]] = {
        node_id: [LSP(**lsp) for lsp in queue]
        for node_id, queue in ls_flood_queues.items()
    }

    return InitializeResponse(
        dvTables=dv_tables_out,
        lsdb=lsdb_out,
        lsFloodQueues=ls_flood_queues_out,
        lspSeqCounter=lsp_seq_counter["value"],
    )


@app.post("/api/routing/step", response_model=StepResponse)
def routing_step(req: StepRequest):
    """
    Runs one simulation step (generate messages + apply routing updates).
    Mirrors generateMessages() + applyMessages() in useNetworkStore.ts.
    """
    nodes = [n.model_dump() for n in req.nodes]
    links = [l.model_dump() for l in req.links]
    protocol = req.protocol
    packet_loss_rate = req.packetLossRate
    packet_dist = req.packetDist
    link_failure_rate = req.linkFailureRate
    link_fail_dist = req.linkFailDist
    link_recover_dist = req.linkRecoverDist
    iteration_count = req.iterationCount

    # Deserialize tables
    dv_tables: Dict[str, Dict[str, dict]] = {
        node_id: {dest_id: dict(entry) for dest_id, entry in table.items()}
        for node_id, table in req.dvTables.items()
    }
    lsdb: Dict[str, Dict[str, dict]] = {
        node_id: {src_id: dict(lsp) for src_id, lsp in db.items()}
        for node_id, db in req.lsdb.items()
    }
    ls_flood_queues: Dict[str, List[dict]] = {
        node_id: [dict(lsp) for lsp in queue]
        for node_id, queue in req.lsFloodQueues.items()
    }
    counter = req.lspSeqCounter

    # ── 1. Network Weather (Link Failure / Recovery) ──────────────────────
    current_links = [dict(l) for l in links]
    new_dv_tables = {k: {dk: dict(dv) for dk, dv in v.items()} for k, v in dv_tables.items()}
    new_lsdb = {k: {sk: dict(lsp) for sk, lsp in v.items()} for k, v in lsdb.items()}
    new_flood_queues = {k: [dict(l) for l in v] for k, v in ls_flood_queues.items()}
    weather_calculations: List[dict] = []
    any_link_toggled = False

    for link in current_links:
        status_changed = False
        event_name = ""
        node_a = link["sourceId"]
        node_b = link["targetId"]

        if link.get("status") == "down":
            down_ticks = (link.get("downTicks") or 0) + 1
            recover_prob = min(1.0, 0.4 + down_ticks * 0.1)
            if check_probability(link_recover_dist, recover_prob):
                link["status"] = "up"
                link["downTicks"] = 0
                any_link_toggled = True
                status_changed = True
                event_name = "Link Up"
            else:
                link["downTicks"] = down_ticks
        else:
            if check_probability(link_fail_dist, link_failure_rate):
                link["status"] = "down"
                link["downTicks"] = 1
                any_link_toggled = True
                status_changed = True
                event_name = "Link Down"

        if status_changed:
            for node_id in [node_a, node_b]:
                neighbor_id = node_b if node_id == node_a else node_a
                if protocol == "DV" and node_id in new_dv_tables:
                    my_table = new_dv_tables[node_id]
                    changed = False
                    if event_name == "Link Down":
                        for entry in my_table.values():
                            if entry.get("nextHopId") == neighbor_id:
                                entry["cost"] = INF
                                changed = True
                    else:
                        if neighbor_id not in my_table or my_table[neighbor_id]["cost"] > link["cost"]:
                            my_table[neighbor_id] = {"destinationId": neighbor_id, "nextHopId": neighbor_id, "cost": link["cost"]}
                            changed = True
                    if changed:
                        weather_calculations.append({"nodeId": node_id, "equation": f"Cable {event_name} → Table Updated"})

                elif protocol == "LS":
                    counter += 1
                    new_lsp = generate_lsp(node_id, counter, current_links)
                    my_new_lsdb = {**(new_lsdb.get(node_id) or {}), node_id: new_lsp}
                    new_lsdb[node_id] = my_new_lsdb
                    existing = new_flood_queues.get(node_id, [])
                    new_flood_queues[node_id] = existing + [new_lsp]
                    new_dv_tables[node_id] = calculate_dijkstra(node_id, my_new_lsdb)
                    weather_calculations.append({"nodeId": node_id, "equation": f"LSP Originated ({event_name}) → SPF Recomputed"})

    # ── 2. Generate Messages for this Step ───────────────────────────────
    messages: List[dict] = []

    if protocol == "DV":
        for node in nodes:
            node_id = node["id"]
            my_table = new_dv_tables.get(node_id)
            if not my_table:
                continue
            connected = [l for l in current_links if (l["sourceId"] == node_id or l["targetId"] == node_id) and l.get("status") == "up"]
            for link in connected:
                target_id = link["targetId"] if link["sourceId"] == node_id else link["sourceId"]
                is_dropped = check_probability(packet_dist, packet_loss_rate) if packet_loss_rate > 0 else False

                # Build payload with Split Horizon / Poison Reverse
                payload: Dict[str, dict] = {}
                for entry in my_table.values():
                    payload[entry["destinationId"]] = {
                        "destinationId": entry["destinationId"],
                        "cost": INF if entry.get("nextHopId") == target_id else entry["cost"],
                    }

                messages.append({
                    "id": str(uuid.uuid4()),
                    "sourceNodeId": node_id,
                    "targetNodeId": target_id,
                    "linkId": link["id"],
                    "type": "DV_UPDATE",
                    "payload": payload,
                    "status": "dropped" if is_dropped else "delivered",
                })

    elif protocol == "LS":
        for node in nodes:
            node_id = node["id"]
            queue = new_flood_queues.get(node_id, [])
            if not queue:
                continue
            connected = [l for l in current_links if (l["sourceId"] == node_id or l["targetId"] == node_id) and l.get("status") == "up"]
            for link in connected:
                target_id = link["targetId"] if link["sourceId"] == node_id else link["sourceId"]
                is_dropped = check_probability(packet_dist, packet_loss_rate) if packet_loss_rate > 0 else False
                messages.append({
                    "id": str(uuid.uuid4()),
                    "sourceNodeId": node_id,
                    "targetNodeId": target_id,
                    "linkId": link["id"],
                    "type": "LSP_FLOOD",
                    "payload": queue,
                    "status": "dropped" if is_dropped else "delivered",
                })
        # Clear all flood queues
        for node in nodes:
            new_flood_queues[node["id"]] = []

    if not messages:
        # No messages → already converged
        return _build_step_response(
            links=current_links,
            dv_tables=new_dv_tables,
            lsdb=new_lsdb,
            ls_flood_queues=new_flood_queues,
            active_messages=[],
            calc_logs=[],
            active_calcs=weather_calculations,
            simulation_state="CONVERGED",
            iteration_count=0,
            lsp_counter=counter,
        )

    # ── 3. Apply Messages ─────────────────────────────────────────────────
    new_logs: List[dict] = []
    new_calculations: List[dict] = list(weather_calculations)
    node_map = {n["id"]: n for n in nodes}

    for msg in messages:
        if msg["status"] == "dropped":
            continue
        target_node = node_map.get(msg["targetNodeId"])
        source_node = node_map.get(msg["sourceNodeId"])
        link = next((l for l in current_links if l["id"] == msg["linkId"]), None)
        if not target_node or not source_node or not link or link.get("status") == "down":
            continue

        if msg["type"] == "DV_UPDATE" and protocol == "DV":
            current_table = new_dv_tables.get(target_node["id"], {})
            new_table, changed, updates_log = process_dv_update_with_logs(
                nodes, current_table, source_node, target_node, msg["payload"], link["cost"]
            )
            new_dv_tables[target_node["id"]] = new_table
            if changed:
                for log_msg in updates_log:
                    new_logs.append({"id": str(uuid.uuid4()), "message": log_msg})
                    import re
                    match = re.search(r"Cost: (.*)", log_msg)
                    if match:
                        new_calculations.append({"nodeId": target_node["id"], "equation": match.group(1)})
                    else:
                        new_calculations.append({"nodeId": target_node["id"], "equation": "Table Updated"})

        elif msg["type"] == "LSP_FLOOD" and protocol == "LS":
            incoming_lsps = msg["payload"]
            my_lsdb = dict(new_lsdb.get(target_node["id"], {}))
            updated = False
            accepted_details: List[str] = []

            for incoming_lsp in incoming_lsps:
                current_lsp = my_lsdb.get(incoming_lsp["sourceId"])
                if not current_lsp or incoming_lsp["sequenceNumber"] > current_lsp["sequenceNumber"]:
                    my_lsdb[incoming_lsp["sourceId"]] = incoming_lsp
                    existing_q = new_flood_queues.get(target_node["id"], [])
                    new_flood_queues[target_node["id"]] = existing_q + [incoming_lsp]
                    src_name = node_map.get(incoming_lsp["sourceId"], {}).get("name", incoming_lsp["sourceId"])
                    edges = " ".join(f"→{node_map.get(l['targetId'], {}).get('name', '?')}({l['cost']})" for l in incoming_lsp.get("links", []))
                    accepted_details.append(f"LSP from {src_name} (Seq {incoming_lsp['sequenceNumber']}): [{edges}]")
                    updated = True

            if updated:
                new_lsdb[target_node["id"]] = my_lsdb
                old_table = new_dv_tables.get(target_node["id"], {})
                new_table = calculate_dijkstra(target_node["id"], my_lsdb)
                new_dv_tables[target_node["id"]] = new_table

                for detail in accepted_details:
                    new_logs.append({"id": str(uuid.uuid4()), "message": f"[{target_node['name']}] Accepted {detail}"})

                for dest_id, new_entry in new_table.items():
                    old_entry = old_table.get(dest_id)
                    if not old_entry or old_entry["cost"] != new_entry["cost"] or old_entry["nextHopId"] != new_entry["nextHopId"]:
                        dest_name = node_map.get(dest_id, {}).get("name", "?")
                        hop_name = "Direct" if new_entry["nextHopId"] == target_node["id"] else node_map.get(new_entry.get("nextHopId", ""), {}).get("name", "-")
                        cost_str = "∞" if new_entry["cost"] > 900000 else str(new_entry["cost"])
                        new_calculations.append({"nodeId": target_node["id"], "equation": f"SPF: Path to {dest_name} → Via {hop_name} (Cost: {cost_str})"})

    # ── 4. Check Convergence ──────────────────────────────────────────────
    max_iterations = max(1, len(nodes) - 1)
    new_iter_count = iteration_count + 1
    if new_iter_count >= max_iterations:
        sim_state = "CONVERGED"
        first_node_id = nodes[0]["id"] if nodes else ""
        new_calculations.append({"nodeId": first_node_id, "equation": f"Algorithm Converged at N-1 Iterations ({max_iterations})"})
        new_iter_count = 0
    else:
        sim_state = "READY"

    return _build_step_response(
        links=current_links,
        dv_tables=new_dv_tables,
        lsdb=new_lsdb,
        ls_flood_queues=new_flood_queues,
        active_messages=messages,
        calc_logs=new_logs,
        active_calcs=new_calculations,
        simulation_state=sim_state,
        iteration_count=new_iter_count,
        lsp_counter=counter,
    )


def _build_step_response(
    links, dv_tables, lsdb, ls_flood_queues,
    active_messages, calc_logs, active_calcs,
    simulation_state, iteration_count, lsp_counter
) -> StepResponse:
    """Helper to assemble a StepResponse from raw dicts."""
    from models import RouterLink

    links_out = [RouterLink(**l) for l in links]
    dv_out = {
        nid: {did: DVEntry(**e) for did, e in table.items()}
        for nid, table in dv_tables.items()
    }
    lsdb_out = {
        nid: {sid: LSP(**l) for sid, l in db.items()}
        for nid, db in lsdb.items()
    }
    queues_out = {
        nid: [LSP(**l) for l in queue]
        for nid, queue in ls_flood_queues.items()
    }
    msgs_out = [RouteMessage(**m) for m in active_messages]
    logs_out = [CalculationLog(**l) for l in calc_logs]
    calcs_out = [ActiveCalculation(**c) for c in active_calcs]

    return StepResponse(
        links=links_out,
        dvTables=dv_out,
        lsdb=lsdb_out,
        lsFloodQueues=queues_out,
        activeMessages=msgs_out,
        calculationLogs=logs_out,
        activeCalculations=calcs_out,
        simulationState=simulation_state,
        iterationCount=iteration_count,
        lspSeqCounter=lsp_counter,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Probability Utility Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/probability/check", response_model=ProbabilityResponse)
def probability_check(req: ProbabilityRequest):
    """Evaluates a single probability check with the given distribution mode."""
    return ProbabilityResponse(result=check_probability(req.mode, req.p))


@app.get("/api/probability/edge-weight")
def probability_edge_weight():
    """Generates a realistic network edge weight using the exponential distribution."""
    return {"weight": generate_realistic_edge_weight()}


# ─────────────────────────────────────────────────────────────────────────────
# ARP Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class ArpNodeRequest(BaseModel):
    node_type: str
    x: float
    y: float
    existing_nodes: List[dict] = []


class ArpRequestPayload(BaseModel):
    source_node_id: str
    target_ip: str
    nodes: List[dict]
    links: List[dict]
    arp_caches: Dict[str, List[dict]] = {}
    tick_counter: int = 0
    packet_loss_rate: float = 0.0
    packet_dist: str = "Uniform"


class ArpTickRequest(BaseModel):
    nodes: List[dict]
    links: List[dict]
    active_packets: List[dict]
    arp_caches: Dict[str, List[dict]] = {}
    switch_mac_tables: Dict[str, List[dict]] = {}
    tick_counter: int = 0
    cache_ttl: int = 300
    packet_loss_rate: float = 0.0
    link_failure_rate: float = 0.0
    packet_dist: str = "Uniform"
    link_fail_dist: str = "Uniform"
    link_recover_dist: str = "Uniform"


@app.post("/api/arp/node")
def arp_create_node(req: ArpNodeRequest):
    """Creates a new ARP topology node."""
    return create_arp_node(req.node_type, req.x, req.y, req.existing_nodes)


@app.post("/api/arp/request")
def arp_send_request(req: ArpRequestPayload):
    """Initiates an ARP request from a source node targeting an IP address."""
    packets, logs = initiate_arp_request(
        req.source_node_id, req.target_ip,
        req.nodes, req.links, req.arp_caches,
        req.tick_counter, req.packet_loss_rate, req.packet_dist,
    )
    return {"packets": packets, "logs": logs}


@app.post("/api/arp/tick")
def arp_simulation_tick(req: ArpTickRequest):
    """Advances the ARP simulation by one tick."""
    return advance_arp_tick(
        req.nodes, req.links, req.active_packets,
        req.arp_caches, req.switch_mac_tables,
        req.tick_counter, req.cache_ttl,
        req.packet_loss_rate, req.link_failure_rate,
        req.packet_dist, req.link_fail_dist, req.link_recover_dist,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DHCP Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class DhcpNodeRequest(BaseModel):
    node_type: str
    x: float
    y: float
    existing_nodes: List[dict] = []
    prng_seed: int = 42


class DoraRequest(BaseModel):
    client_id: str
    nodes: List[dict]
    links: List[dict]
    dora_sequences: List[dict] = []
    leases: List[dict] = []
    tick_counter: int = 0
    prng_seed: int = 42
    packet_loss_rate: float = 0.0


class MonteCarloRequest(BaseModel):
    total_runs: int = 1000
    packet_loss_percent: float = 10.0
    max_retries: int = 3
    num_clients: int = 1


@app.post("/api/dhcp/node")
def dhcp_create_node(req: DhcpNodeRequest):
    """Creates a new DHCP topology node."""
    return create_dhcp_node(req.node_type, req.x, req.y, req.existing_nodes, req.prng_seed)


@app.post("/api/dhcp/dora/start")
def dhcp_start_dora(req: DoraRequest):
    """Initiates a DORA sequence for a DHCP client."""
    return start_dora(
        req.client_id, req.nodes, req.links,
        req.dora_sequences, req.leases,
        req.tick_counter, req.prng_seed, req.packet_loss_rate,
    )


@app.post("/api/dhcp/montecarlo")
def dhcp_monte_carlo(req: MonteCarloRequest):
    """Runs a Monte Carlo simulation of the DHCP DORA process."""
    return run_monte_carlo(
        req.total_runs, req.packet_loss_percent,
        req.max_retries, req.num_clients,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fragmentation Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class FragNodeRequest(BaseModel):
    node_type: str
    x: float
    y: float
    existing_nodes: List[dict] = []


class FragSendRequest(BaseModel):
    source_id: str
    target_id: str
    payload_size: int
    df_flag: int = 0
    links: List[dict]
    packet_counter: int = STARTING_IDENTIFICATION


class FragTickRequest(BaseModel):
    nodes: List[dict]
    links: List[dict]
    active_packets: List[dict] = []
    reassembly_buffers: List[dict] = []
    drop_animations: List[dict] = []
    tick_counter: int = 0
    simulation_speed: float = 1.0
    reassembly_timeout: int = 200


@app.post("/api/fragmentation/node")
def frag_create_node(req: FragNodeRequest):
    """Creates a new fragmentation topology node."""
    return create_frag_node(req.node_type, req.x, req.y, req.existing_nodes)


@app.post("/api/fragmentation/send")
def frag_send_packet_endpoint(req: FragSendRequest):
    """Creates IP packet(s) — fragmenting at source if needed."""
    packets, new_counter = frag_send_packet(
        req.source_id, req.target_id, req.payload_size,
        req.df_flag, req.links, req.packet_counter,
    )
    return {"packets": packets, "packetCounter": new_counter}


@app.post("/api/fragmentation/tick")
def frag_simulation_tick_endpoint(req: FragTickRequest):
    """Advances the fragmentation simulation by one tick."""
    return frag_tick(
        req.nodes, req.links, req.active_packets,
        req.reassembly_buffers, req.drop_animations,
        req.tick_counter, req.simulation_speed, req.reassembly_timeout,
    )
