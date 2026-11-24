# Geo-Miner SERI V3

Geo-Miner SERI V3 is a multi-modal geospatial mineral exploration risk and feasibility engine built for Karnataka. It fuses 2D Convolutional Neural Network (CNN) feature extraction over 64x64 satellite nighttime light raster patches with Graph Attention Network (GAT / GATv2) spatial message passing across a regional survey graph of 30,062 nodes and 181,112 edges. The system pairs a FastAPI inference microservice with an interactive Streamlit command dashboard, explainable AI (Grad-CAM) visual overlays, blind holdout validation tools, and a SERI-scoped AI assistant.

---

## Table of Contents

1. [System Architecture & Capabilities](#system-architecture--capabilities)
2. [Tech Stack](#tech-stack)
3. [Project Directory Structure](#project-directory-structure)
4. [What Each File and Module Does](#what-each-file-and-module-does)
5. [Data & Model Artifacts](#data--model-artifacts)
6. [Execution Instructions (Step-by-Step)](#execution-instructions-step-by-step)
   - [Prerequisites](#prerequisites)
   - [Step 1: Clone & Enter Repository](#step-1-clone--enter-repository)
   - [Step 2: Create & Activate Virtual Environment](#step-2-create--activate-virtual-environment)
   - [Step 3: Install Dependencies](#step-3-install-dependencies)
   - [Step 4: Verify Data & Patch Files](#step-4-verify-data--patch-files)
   - [Step 5: Configure Environment Variables](#step-5-configure-environment-variables)
   - [Step 6: Start the FastAPI Microservice](#step-6-start-the-fastapi-microservice)
   - [Step 7: Launch the Streamlit Dashboard](#step-7-launch-the-streamlit-dashboard)
   - [Step 8: Run Automated Tests](#step-8-run-automated-tests)
7. [Dashboard Tabs & Operational Guide](#dashboard-tabs--operational-guide)
8. [Metrics, Risk Tiers & Glossary](#metrics-risk-tiers--glossary)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## System Architecture & Capabilities

- **Dual-Branch Late-Fusion Model**: Processes visual raster patches (`64x64`) through a 2D-CNN branch while aggregating spatial neighborhood context through a Graph Attention Network (GAT / GATv2) across Karnataka's spatial graph (`30,062` nodes, `181,112` edges).
- **Macro Prospecting Map**: Ranks surveyed coordinates across Karnataka by `SERI_Target` risk, isolates configurable numbers of prime blocks, synchronizes map point selection with a node selector dropdown, resolves nearest reference anchors across Karnataka, and explains the primary risk driver for each node.
- **Micro Live Inference Engine**: Accepts 5-D analytical risk/feasibility sliders (`Logistics Friction`, `Cultural Friction`, `Natural Constraints`, `Economic Proximity`, `Engine 1 Mineral Score`), target graph node ID (`0` to `30061`), and terrain topology selection, sends a JSON payload to `[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)`, and visualizes tier classification, connected GAT neighbors, confidence, and Grad-CAM heatmaps.
- **Validation & Volatility Stress Testing**: Enforces a `70/15/15` train/validation/test node masking split, samples unseen nodes from the blind test mask, injects random factor shocks, and displays convergence loss curves and confusion matrix diagnostics.
- **Satellite Feature & Radiance Inspection**: Displays `karnataka_nightlight.tif` full-raster context with node pixel markers, `64x64` patch radiance histograms, multi-factor node attributes, and live `SERI_Predictor` outputs.
- **Domain-Scoped AI Assistant**: Integrates `google-genai` (`gemini-2.5-flash`) in the sidebar with strict system instructions limited to SERI V3 architecture, GAT mechanics, Grad-CAM XAI, and SERI risk metrics.

---

## Tech Stack

- **Programming Language**: Python 3.11+
- **Deep Learning & Graph Neural Networks**: PyTorch (`torch`), PyTorch Geometric (`torch-geometric`), GAT / GATv2 message passing
- **Scientific & Numerical Computing**: NumPy (`numpy`), SciPy (`scipy.ndimage`), Pandas (`pandas`)
- **Explainable AI (XAI)**: Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Backend API & Validation**: FastAPI (`fastapi`), Uvicorn (`uvicorn`), Pydantic (`pydantic`), HTTP client (`requests`)
- **Frontend & Data Visualization**: Streamlit (`streamlit`), Plotly (`plotly.express`, `plotly.graph_objects`), Matplotlib (`matplotlib.pyplot`), Pillow (`PIL`)
- **Generative AI Assistant**: Google GenAI SDK (`google-genai`, model `gemini-2.5-flash`)
- **Testing & Source Control**: `pytest`, Git, GitHub

---

## Project Directory Structure

```text
SERI_Core/
├── .gitignore
├── README.md
├── requirements.txt
├── api/
│   └── main.py
├── core/
│   ├── __init__.py
│   └── inference.py
├── dashboard/
│   └── app.py
├── data/
│   ├── karnataka_master_data.csv
│   ├── karnataka_graph.pt
│   └── karnataka_nightlight.tif
├── seri_patches/
│   └── patch_<node_id>.npy
└── tests/
    ├── test_api.py
    └── test_inference.py
```

---

## What Each File and Module Does

- **`dashboard/app.py`**: Main Streamlit application. Configures page styling, typography, and color tokens; loads and caches `karnataka_master_data.csv` and `karnataka_nightlight.tif`; renders the sidebar command hub, target count slider, tier legend, and Gemini chat assistant; and implements all five main tabs (`State-Wide Prospecting (Macro)`, `Live Inference Engine (Micro)`, `System Validation & Stress Test`, `Satellite Feature Display`, and `System Documentation & Metric Guide`).
- **`api/main.py`**: FastAPI microservice entrypoint. Exposes `POST /predict` (accepting `vectors`, `patch_flat`, and `node_idx`) and invokes the inference pipeline to return classification tier, confidence score, connected neighbor/edge counts, and Grad-CAM activation arrays.
- **`core/inference.py`**: Core inference engine (`SERI_Predictor`). Loads model weights, graph tensors (`karnataka_graph.pt`), and tabular data (`karnataka_master_data.csv`), executes 2D-CNN + GAT/GATv2 forward passes, generates Grad-CAM heatmaps/overlays, and returns structured prediction dictionaries.
- **`data/karnataka_master_data.csv`**: Master tabular dataset for surveyed points in Karnataka, containing coordinates (`Latitude`, `Longitude`), risk and feasibility columns (`Logistics_Risk`, `Cult_Risk`, `Nature_Risk`, `Econ_Risk`), and target values (`SERI_Target`).
- **`data/karnataka_graph.pt`**: PyTorch Geometric graph object containing node feature tensors (`x`), graph connectivity (`edge_index`), and boolean split masks (`train_mask`, `val_mask`, `test_mask`).
- **`data/karnataka_nightlight.tif`**: Nighttime satellite raster dataset covering Karnataka used for full-raster visualization and coordinate-to-pixel mapping.
- **`seri_patches/`**: Directory of pre-saved `64x64` NumPy array patches (`patch_<node_id>.npy`) representing localized nighttime/terrain patches per graph node, with procedural/Gaussian fallback generation when a specific `.npy` patch is absent.
- **`tests/`**: Automated `pytest` test suite covering API request/response contracts, inference outputs, and data pipeline sanity checks.
- **`requirements.txt`**: Pinned/minimum Python package dependencies required to run the API, inference engine, dashboard, and test suite.
- **`.gitignore`**: Excludes virtual environments (`.venv/`, `venv/`), Python bytecode (`__pycache__/`, `*.pyc`), environment files (`.env`), logs, and OS artifacts (`.DS_Store`).

---

## Data & Model Artifacts

Ensure the following files are present in your working tree before running full live predictions:

1. `data/karnataka_master_data.csv`
2. `data/karnataka_graph.pt`
3. `data/karnataka_nightlight.tif`
4. `seri_patches/` (containing `64x64` `.npy` patches)
5. Trained model checkpoint / weight files expected by `core/inference.py` and `api/main.py`.

---

## Execution Instructions (Step-by-Step)

### Prerequisites

- **Python**: Version `3.11` or higher installed and available on `PATH` (`python3 --version`).
- **Git**: Installed (`git --version`) for cloning and version control.
- **Ports**: Port `8000` free for the FastAPI backend and port `8501` free for the Streamlit UI.
- **Gemini API Key (Optional)**: Required only if you want to use the sidebar **SERI AI Assistant** tab/panel.

### Step 1: Clone & Enter Repository

```bash
git clone https://github.com/<your-username>/SERI_Core.git
cd SERI_Core
```

### Step 2: Create & Activate Virtual Environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: If PyTorch or PyTorch Geometric requires a specific wheel index for your OS, CPU/MPS, or CUDA version, install `torch` and `torch-geometric` per the official PyTorch / PyG installation matrix before or after `pip install -r requirements.txt`.

### Step 4: Verify Data & Patch Files

Confirm your expected data files exist from the `SERI_Core` root directory:

```bash
ls -la data/ seri_patches/ api/ core/ dashboard/
```

### Step 5: Configure Environment Variables

If you have a Google Gemini API key for the sidebar assistant, export it in your shell:

**macOS / Linux:**

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

Alternatively, you can leave `GEMINI_API_KEY` unset and enter the key directly in the sidebar password field when the dashboard loads.

### Step 6: Start the FastAPI Microservice

Open **Terminal 1**, make sure `.venv` is activated and you are in `SERI_Core`, and start Uvicorn on port `8000`:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Keep this terminal running. You can verify the API schema and health in a browser at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

### Step 7: Launch the Streamlit Dashboard

Open **Terminal 2**, activate the same `.venv`, change into `SERI_Core`, and start Streamlit:

```bash
streamlit run dashboard/app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Open it in your browser.

### Step 8: Run Automated Tests

In a terminal with `.venv` active inside `SERI_Core`:

```bash
pytest -v
```

---

## Dashboard Tabs & Operational Guide

1. **State-Wide Prospecting (Macro)**:
   - Adjust **Number of Prime Blocks to Isolate** in the sidebar (`10` to `2000`, default `1760`).
   - Select a node from the **Conjoined Point Selector** dropdown or click a point directly on the Plotly map to inspect its coordinates, nearest Karnataka reference city and distance, risk tier, and top risk driver.
2. **Live Inference Engine (Micro)**:
   - Set **Logistics Friction**, **Cultural Friction**, **Natural Constraints**, **Economic Proximity**, and **Engine 1 Mineral Score** (`0.0` to `1.0`).
   - Choose a simulated raster block topology and a **Target Graph Node ID** (`0` to `30061`).
   - Click **Run Live Multi-Modal Scan** to send the payload to `[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)` and inspect the predicted tier, connected GAT neighbors, confidence bar, and 3-panel Grad-CAM visualization (`Input Raster`, `Gradient Activation (Grad-CAM)`, `Fused Overlay`).
3. **System Validation & Stress Test**:
   - Review the `70/15/15` node masking protocol (`70%` train, `15%` validation, `15%` blind test).
   - Click **Sample Unseen Target Node** to draw a node from `test_mask` in `karnataka_graph.pt` and inspect its normalized feature vector.
   - Click **Inject Random Factor Shocks** to generate random factor deltas (`-0.40` to `+0.40`).
   - Inspect the learning convergence curve and blind test confusion matrix.
4. **Satellite Feature Display**:
   - Use **Pull Random Prime Sample**, **Pull Random Marginal Sample**, or **Pull Random Borderline Sample**, or enter a **Graph Node ID** directly.
   - View nighttime radiance regime classification, multi-factor attributes (`Logistics_Risk`, `Cult_Risk`, `Nature_Risk`, `Econ_Risk`, `SERI_Target`), full-raster context with node crosshair, `64x64` patch image, pixel radiance histogram, and click **Run Nighttime + GATv2 Minability Readout** for live `SERI_Predictor` probabilities and Grad-CAM overlay.
5. **System Documentation & Metric Guide**:
   - Read the architecture summary, XAI explanation, plain-language metric glossary, and 3-stage pipeline flow (`Data ingestion` -> `Spatial processing` -> `Verified output`).

---

## Metrics, Risk Tiers & Glossary

- **SERI Score (`SERI_Target`)**: Composite risk/feasibility score where lower values indicate lower friction / higher viability and higher values indicate greater extraction constraints.
- **Prime**: Top-viability tier with minimal friction for extraction.
- **Marginal**: Mid-tier zone that is potentially viable but warrants an environmental or logistical audit.
- **Borderline**: Lower-tier zone associated with low yield or stronger natural, logistical, or cultural constraints.
- **GAT Connected Neighbors**: Number of graph neighbors / edges connected to a target node during Graph Attention message passing.
- **Grad-CAM Heatmap**: Gradient-weighted Class Activation Mapping highlighting spatial regions within a raster patch that most influence the model's prediction.

---

## Troubleshooting Common Issues

- **`Connection Timed Out` in Live Inference Engine (Micro)**: Confirm `uvicorn api.main:app --host 127.0.0.1 --port 8000` is running in a separate terminal and not blocked by a firewall or port conflict.
- **Data file not found (`karnataka_master_data.csv`, `karnataka_graph.pt`, `karnataka_nightlight.tif`)**: Run `streamlit run dashboard/app.py` and `uvicorn api.main:app` from the `SERI_Core` repository root so relative `data/` paths resolve properly.
- **Sidebar AI Assistant disabled**: Set `GEMINI_API_KEY` in your environment before launching Streamlit or paste your API key into the sidebar input box.
- **Torch / PyTorch Geometric import errors**: Verify that `torch` and `torch_geometric` are installed inside the active `.venv` matching your Python and hardware setup.
