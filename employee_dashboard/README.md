# EV Charge Advisor - Employee Dashboard MVP

This is a Streamlit MVP for the AI Hackathon employee dashboard.

## What it does

- Office selector
- Day/time selector
- Forecasted congestion by hour
- Recommended charging window
- Estimated availability probability
- AI-style explanation for employees

## How to run

```bash
cd ev_charge_advisor
pip install streamlit pandas plotly
streamlit run app.py
```

## Data

The app uses `chargepoint_sessions.csv`, copied from the ChargePoint YTD session export.

## Important note

Availability probability is estimated from historical congestion patterns. It is not live charger availability.
