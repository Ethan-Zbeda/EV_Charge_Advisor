# EV Charge Advisor – Leadership Dashboard

A Streamlit dashboard for strategic oversight of workplace EV charging infrastructure. Built for the AI Hackathon.

## Features

### Fleet overview KPIs
Eight metric cards with month-over-month delta indicators:
- Total sessions & energy delivered (with MoM change arrows)
- Unique users & estimated charging ports
- Average energy per session & average session duration
- **Estimated fleet electricity cost** (adjustable rate in sidebar)
- **Estimated CO₂ avoided** vs average ICE vehicle (adjustable factor in sidebar)

### Utilization by office
Bar charts showing total charging sessions and energy delivered per office, colour-scaled for instant comparison.

### Monthly demand trend
Per-office line charts for sessions and energy over time, plus a **fleet-wide growth & 3-month projection** chart with a linear trendline and projected bars. Includes a plain-English growth rate callout (e.g. "growing at 4.2% MoM").

### Sustainability impact
Four KPI cards derived from the sidebar assumptions:
- CO₂ avoided (tonnes)
- Tree-years equivalent (21 kg CO₂/tree/year)
- Gasoline offset (litres)
- Estimated total fleet charging cost ($)

### Average charger utilization rate
Colour-coded bar chart (🟢 Healthy / 🟠 Moderate / 🔴 Critical) showing average charger occupancy during business hours (6 AM – 8 PM).

### Charging efficiency – idle time analysis
Two charts highlighting where vehicles occupy chargers after charging is complete:
- Average idle time per session by office
- Total charger-hours wasted by office

High idle time is a key lever for improving availability without adding hardware.

### Demand heatmap
Day-of-week × hour heatmap per office (selectable). Uses a Red–Yellow–Green colour scale to show when chargers are under the most pressure.

### Peak demand hours
Grouped bar chart comparing average hourly utilization across all selected offices.

### Top power users
Anonymized profiles (User …1234) showing:
- Top 10 users by total energy consumed
- Top 10 users by session count

### Infrastructure planning summary
Sortable table with sessions, unique users, energy, charger ports, utilization %, status, and sessions-per-charger for every office.

### AI summary
Auto-generated strategic narrative covering:
- Fleet-wide totals
- Critical / Moderate / Healthy office classification
- Sustainability impact (CO₂ + cost)
- Idle-time efficiency opportunity (if significant)
- Infrastructure expansion recommendations

## Sidebar controls

| Control | Default | Purpose |
|---|---|---|
| Offices | All | Filter all charts and KPIs |
| Months | All | Filter by calendar month |
| Electricity cost ($/kWh) | $0.15 | Fleet cost estimate |
| CO₂ saved per kWh (kg) | 0.7 | Sustainability impact estimate |

## How to run

```bash
cd leadership_dashboard
pip install streamlit pandas plotly numpy
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

## Data

Uses `chargepoint_sessions.csv` — the ChargePoint YTD session export. The file should be placed in the `leadership_dashboard/` folder.

## Important notes

- Utilization rates are estimated from **historical occupancy patterns**, not live charger telemetry.
- Charger port count is estimated from distinct Station Name + Port Number combinations in the export.
- The 3-month projection uses a simple linear trend and should be treated as directional, not precise.
- CO₂ and cost figures are estimates based on the sidebar assumptions.
