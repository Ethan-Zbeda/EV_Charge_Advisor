# EV Charge Advisor – Leadership Dashboard

A Streamlit dashboard for strategic oversight of workplace EV charging infrastructure. Built for the AI Hackathon.

## Features

### Fleet overview KPIs
Six metric cards giving an instant fleet-wide snapshot:
- Total charging sessions & total energy delivered (kWh)
- Unique EV drivers & estimated charging ports
- Average energy per session (kWh) & average session duration (hrs)

### Utilization by office
Side-by-side bar charts showing total charging sessions and total energy delivered per office, styled in NextEra brand colors.

### Monthly demand trend
Per-office line charts for session volume and energy delivered over time, making it easy to spot growth patterns and seasonal shifts across locations.

### Average charger utilization rate
Color-coded bar chart showing average charger occupancy during business hours (6 AM – 8 PM):
- 🟢 **Healthy** — below 45 % utilization
- 🟠 **Moderate** — 45 – 70 % utilization
- 🔴 **Critical** — above 70 % (infrastructure expansion recommended)

### Demand heatmap
Selectable per-office day-of-week × hour heatmap showing when chargers are under the most pressure, using the NextEra blue color scale.

### Peak demand hours
Grouped bar chart comparing average hourly utilization rate across all selected offices during business hours.

### Infrastructure planning summary
Sortable table with sessions, unique users, energy (kWh), charger ports, average utilization %, status, and sessions-per-charger for every office — designed to support data-driven expansion decisions.

### AI Assistant
Streaming chat assistant powered by a local [Ollama](https://ollama.com) model (same backend as the Employee Dashboard). The assistant has full context of the current fleet snapshot — total sessions, energy, utilization by office — and can answer questions about trends, capacity planning, or infrastructure recommendations in plain language.

## Sidebar controls

| Control | Default | Purpose |
|---|---|---|
| Offices | All | Filter all charts, KPIs, and AI context |
| Months | All | Filter by calendar month |

## How to run

```bash
cd leadership_dashboard
pip install streamlit pandas plotly requests
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

For the AI Assistant, Ollama must be running locally:

```bash
ollama serve
ollama pull llama3.2
```

## Data

Uses `chargepoint_sessions.csv` — the ChargePoint YTD session export. Place the file in the `leadership_dashboard/` folder.

## Important notes

- Utilization rates are estimated from **historical occupancy patterns**, not live charger telemetry.
- Charger port count is estimated from distinct Station Name + Port Number combinations in the export.
- A utilization rate above 70 % is a strong signal that additional infrastructure is needed at that location.
- The AI Assistant uses a local model via Ollama — no data leaves the machine.
