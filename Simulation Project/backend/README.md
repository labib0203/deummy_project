# Network Simulation — Python Backend

This directory contains the Python backend for the Network Protocol Simulator. It exposes a **FastAPI** REST API that implements all four simulation engines previously written in TypeScript.

## Project Structure

```
backend/
├── main.py                    # FastAPI app & all REST endpoints
├── models.py                  # Pydantic request/response models (mirror of TypeScript interfaces)
├── probability.py             # Probability distribution functions (Uniform, Normal, Binomial, Poisson)
├── distance_vector.py         # Bellman-Ford DV routing algorithm (port of DistanceVector.ts)
├── link_state.py              # Dijkstra LS routing + LSP generation (port of LinkState.ts)
├── arp_simulation.py          # ARP simulation engine (port of useArpStore.ts)
├── dhcp_simulation.py         # DHCP/DORA simulation engine (port of useDhcpStore.ts)
├── fragmentation_simulation.py # IP Fragmentation engine (port of useFragmentationStore.ts)
└── requirements.txt           # Python dependencies
```

## Tech Stack

| Layer       | Technology            |
|-------------|-----------------------|
| Framework   | FastAPI               |
| Server      | Uvicorn (ASGI)        |
| Validation  | Pydantic v2           |
| Language    | Python 3.10+          |

## Setup & Running

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

### 3. Start the frontend (separate terminal)

```bash
cd ..   # back to "Simulation Project" root
npm run dev
```

## API Endpoints

### Routing (Distance Vector + Link State)

| Method | Endpoint                    | Description                            |
|--------|-----------------------------|----------------------------------------|
| POST   | `/api/routing/initialize`   | Initialize DV or LS routing tables     |
| POST   | `/api/routing/step`         | Run one simulation tick (generate + apply messages) |

### ARP

| Method | Endpoint               | Description                              |
|--------|------------------------|------------------------------------------|
| POST   | `/api/arp/node`        | Create a new ARP node (host/switch/router) |
| POST   | `/api/arp/request`     | Initiate an ARP request                  |
| POST   | `/api/arp/tick`        | Advance ARP simulation by one tick       |

### DHCP

| Method | Endpoint                  | Description                          |
|--------|---------------------------|--------------------------------------|
| POST   | `/api/dhcp/node`          | Create a new DHCP node               |
| POST   | `/api/dhcp/dora/start`    | Start a DORA sequence for a client   |
| POST   | `/api/dhcp/montecarlo`    | Run a Monte Carlo DORA simulation    |

### Fragmentation

| Method | Endpoint                    | Description                          |
|--------|-----------------------------|--------------------------------------|
| POST   | `/api/fragmentation/node`   | Create a new fragmentation node      |
| POST   | `/api/fragmentation/send`   | Send a packet (may fragment at source) |
| POST   | `/api/fragmentation/tick`   | Advance fragmentation simulation by one tick |

### Utilities

| Method | Endpoint                       | Description                          |
|--------|--------------------------------|--------------------------------------|
| POST   | `/api/probability/check`       | Check probability with a given distribution |
| GET    | `/api/probability/edge-weight` | Generate a realistic link cost       |

## Module Mapping

| Python File                   | TypeScript Source               |
|-------------------------------|---------------------------------|
| `distance_vector.py`          | `src/simulation/DistanceVector.ts` |
| `link_state.py`               | `src/simulation/LinkState.ts`   |
| `probability.py`              | `src/store/useNetworkStore.ts` (checkProbability) |
| `arp_simulation.py`           | `src/store/useArpStore.ts`      |
| `dhcp_simulation.py`          | `src/store/useDhcpStore.ts`     |
| `fragmentation_simulation.py` | `src/store/useFragmentationStore.ts` |
| `models.py`                   | `src/simulation/Models.ts` + `DhcpModels.ts` + `ArpModels.ts` |
