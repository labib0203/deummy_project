# NetSim — Network Protocol Simulator

A full-stack interactive network simulation platform built with **React** on the frontend and **Python (FastAPI)** on the backend. NetSim allows users to visually construct network topologies and simulate core networking protocols in real time.

---

## 🧱 Tech Stack

| Layer | Technology |
|------------|----------------------------------------|
| Frontend | React, TypeScript, Vite |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Styling | CSS Modules, Custom Animations |
| State Mgmt | React Hooks / Context |
| API | RESTful HTTP (JSON) |

---

## 📦 Project Structure

```
Simulation Project/
├── backend/                  # Python FastAPI backend
│   ├── main.py               # FastAPI app & all API route handlers
│   ├── models.py             # Pydantic request/response models
│   ├── distance_vector.py    # Distance Vector routing algorithm
│   ├── link_state.py         # Link State (Dijkstra) routing algorithm
│   ├── arp_simulation.py     # ARP protocol simulation logic
│   ├── dhcp_simulation.py    # DHCP protocol simulation logic
│   ├── fragmentation_simulation.py  # IP fragmentation logic
│   ├── probability.py        # Packet loss / link reliability utilities
│   ├── requirements.txt      # Python dependencies
│   └── __init__.py
│
├── src/                      # React TypeScript frontend
│   ├── main.tsx              # App entry point
│   ├── App.tsx               # Root component & routing
│   ├── index.css             # Global styles
│   ├── components/           # Reusable UI components
│   │   ├── MainMenu.tsx
│   │   ├── TopologyBuilder.tsx
│   │   ├── RouterNode.tsx
│   │   ├── RouterPalette.tsx
│   │   ├── RoutingTableUI.tsx
│   │   ├── LoadingScreen.tsx
│   │   ├── ProbabilityControls.tsx
│   │   ├── arp/              # ARP-specific UI components
│   │   ├── dhcp/             # DHCP-specific UI components
│   │   └── fragmentation/    # Fragmentation-specific UI components
│   └── simulation/           # TypeScript type models (frontend-side)
│       ├── Models.ts
│       ├── DistanceVector.ts
│       ├── LinkState.ts
│       ├── ArpModels.ts
│       └── DhcpModels.ts
│
├── index.html                # HTML entry point
├── vite.config.ts            # Vite configuration
├── package.json              # Node.js dependencies
└── tsconfig.json             # TypeScript configuration
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+ and npm
- **Python** 3.11+
- `pip` or a virtual environment manager (e.g., `venv`, `conda`)

---

### 1. Start the Python Backend

```bash
# Navigate to the backend folder
cd "Simulation Project/backend"

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload --port 8000
```

The backend API will be available at: `http://localhost:8000`  
Interactive API docs (Swagger UI): `http://localhost:8000/docs`

---

### 2. Start the React Frontend

```bash
# From the project root
cd "Simulation Project"

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at: `http://localhost:5173`

---

## 🌐 Simulated Protocols

| Protocol | Description |
|---|---|
| **Distance Vector** | Bellman-Ford based routing with link-failure propagation |
| **Link State** | Dijkstra's algorithm for shortest-path computation |
| **ARP** | Address Resolution Protocol — MAC/IP table simulation |
| **DHCP** | Dynamic Host Configuration Protocol — IP lease simulation |
| **IP Fragmentation** | MTU-based packet fragmentation and reassembly |

---

## 🔌 API Overview

All simulation logic is handled by the Python backend and exposed via a REST API. The React frontend communicates with these endpoints to drive the UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/distance-vector/compute` | Run DV routing simulation |
| `POST` | `/link-state/compute` | Run LS routing simulation |
| `POST` | `/arp/simulate` | Simulate ARP resolution |
| `POST` | `/dhcp/simulate` | Simulate DHCP lease lifecycle |
| `POST` | `/fragmentation/simulate` | Simulate IP packet fragmentation |

> Full interactive documentation is available via Swagger UI at `http://localhost:8000/docs` when the backend is running.

---

## 🛠️ Development Notes

- The frontend uses **Vite** for fast HMR (Hot Module Replacement) during development.
- The backend uses **FastAPI** with **Pydantic** models for request validation and serialization.
- CORS is enabled on the backend to allow cross-origin requests from the React dev server.
- TypeScript type models in `src/simulation/` mirror the Python Pydantic schemas for type safety end-to-end.

---

## 📄 License

This project is for educational and academic purposes.
