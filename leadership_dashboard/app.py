import os
import json

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EV Charge Advisor", layout="wide")

# NextEra Energy brand palette (sourced from nexteraenergy.com).
NEXTERA_NAVY = "#0c2739"
NEXTERA_BLUE = "#0073A8"
NEXTERA_BLUE_LIGHT = "#1484ba"
NEXTERA_GREEN = "#447b2d"
NEXTERA_RED = "#a94442"
CONGESTION_COLORS = {
    "Low": NEXTERA_GREEN,
    "Medium": NEXTERA_BLUE,
    "High": NEXTERA_RED,
}

st.markdown(
    f"""
    <style>
      /* ── NextEra-branded header band ───────────────────────────── */
      .nee-header {{
        background: linear-gradient(100deg, {NEXTERA_NAVY} 0%, {NEXTERA_BLUE} 100%);
        border-bottom: 4px solid {NEXTERA_GREEN};
        border-radius: 8px;
        padding: 22px 28px;
        margin-bottom: 26px;
      }}
      .nee-header .nee-logo {{
        color: #ffffff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0.92;
      }}
      .nee-header .nee-logo span {{ color: {NEXTERA_GREEN}; }}
      .nee-header h1 {{
        color: #ffffff;
        font-size: 2.05rem;
        font-weight: 700;
        margin: 6px 0 4px 0;
        padding: 0;
      }}
      .nee-header p {{
        color: #d6e6ef;
        font-size: 0.98rem;
        margin: 0;
      }}

      /* ── Section headings: navy with a green accent bar ─────────── */
      .block-container h2, .block-container h3 {{
        color: {NEXTERA_NAVY};
        border-left: 4px solid {NEXTERA_GREEN};
        padding-left: 10px;
      }}

      /* ── Metric cards ──────────────────────────────────────────── */
      [data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #dbe5ec;
        border-top: 3px solid {NEXTERA_BLUE};
        border-radius: 6px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(12, 39, 57, 0.07);
      }}
      [data-testid="stMetricValue"] {{ color: {NEXTERA_BLUE}; }}

      /* ── Sidebar heading accent ────────────────────────────────── */
      [data-testid="stSidebar"] h2 {{
        color: {NEXTERA_NAVY};
        border-left: 4px solid {NEXTERA_GREEN};
        padding-left: 10px;
      }}

      /* ── AI explanation (info box) in NextEra blue ─────────────── */
      [data-testid="stAlert"] {{
        border-left: 5px solid {NEXTERA_BLUE};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "chargepoint_sessions.csv"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_hms(value):
    """Parse a 'hh:mm:ss' duration string into a Timedelta. Hours may exceed 24."""
    try:
        parts = str(value).strip().split(":")
        if len(parts) != 3:
            return pd.NaT
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return pd.Timedelta(hours=h, minutes=m, seconds=s)
    except (ValueError, TypeError):
        return pd.NaT


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["start_dt"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["End Date"], errors="coerce")
    df = df.dropna(subset=["start_dt"])
    df["office"] = df["City"].fillna("Unknown") + " - " + df["Address 1"].fillna("Unknown")
    df["station"] = df["Station Name"].astype(str)
    # Capacity unit = a physical charging port. Each station has multiple ports
    # (Port Number 1, 2, ...), and EVSE ID identifies only the station, so the true
    # port identity is Station Name + Port Number.
    df["port_id"] = df["Station Name"].astype(str) + " / port " + df["Port Number"].astype(str)
    df["user_id"] = df["User ID"]
    df["energy_kwh"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce").fillna(0)

    # Spot occupancy is driven by how long the car is plugged in (Total Duration),
    # not how long it actually charges. A finished car still blocks the charger.
    df["total_duration"] = df["Total Duration (hh:mm:ss)"].apply(parse_hms)
    df["charging_time"] = df["Charging Time (hh:mm:ss)"].apply(parse_hms)

    # Occupancy window = [start, start + total_duration]. Fall back to End Date,
    # then charging time, then a 1-hour default if duration data is missing/invalid.
    dur = df["total_duration"]
    occ_end = df["start_dt"] + dur
    fallback_end = df["end_dt"]
    occ_end = occ_end.where(dur.notna() & (dur > pd.Timedelta(0)), fallback_end)
    still_bad = occ_end.isna() | (occ_end <= df["start_dt"])
    occ_end = occ_end.where(~still_bad, df["start_dt"] + df["charging_time"])
    still_bad = occ_end.isna() | (occ_end <= df["start_dt"])
    occ_end = occ_end.where(~still_bad, df["start_dt"] + pd.Timedelta(hours=1))

    df["occ_start"] = df["start_dt"]
    df["occ_end"] = occ_end
    return df


@st.cache_data
def build_occupancy_table(df):
    """Expand every session across the hours it occupies a charger, then average
    concurrent occupancy across all observed dates for each office/day/hour."""
    records = []
    for r in df.itertuples(index=False):
        start, end = r.occ_start, r.occ_end
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        cur = start.floor("h")
        guard = 0
        while cur < end and guard < 240:  # 240h safety cap for bad rows
            hour_start = cur
            hour_end = cur + pd.Timedelta(hours=1)
            overlap = (min(end, hour_end) - max(start, hour_start)).total_seconds()
            if overlap > 0:
                records.append((
                    r.office,
                    cur.normalize(),       # calendar date (for averaging)
                    cur.dayofweek,
                    cur.hour,
                    overlap / 3600.0,      # fraction of this hour the spot is occupied
                ))
            cur = hour_end
            guard += 1

    occ = pd.DataFrame(
        records,
        columns=["office", "date", "day_num", "hour", "occ_frac"],
    )

    # Sum of occupancy fractions within an hour on a specific date = average number
    # of chargers simultaneously in use during that hour that day.
    per_date = occ.groupby(["office", "date", "day_num", "hour"], as_index=False).agg(
        occupied=("occ_frac", "sum"),
    )

    # Average across all observed dates to get a typical day-of-week / hour profile.
    table = per_date.groupby(["office", "day_num", "hour"], as_index=False).agg(
        occupied=("occupied", "mean"),
    )
    table["day_name"] = table["day_num"].map(dict(enumerate(DAYS)))

    # Capacity = number of distinct charging ports ever seen at the office.
    stations = df.groupby("office")["port_id"].nunique().rename("estimated_chargers")
    table = table.merge(stations, on="office", how="left")
    return table


def build_hourly_table(occ_table, office):
    filtered = occ_table[occ_table["office"] == office].copy()
    stations = int(max(1, filtered["estimated_chargers"].max() if len(filtered) else 1))

    # Full day/hour grid so quiet windows still appear (as fully available).
    grid = pd.MultiIndex.from_product([range(7), range(24)], names=["day_num", "hour"]).to_frame(index=False)
    grid["day_name"] = grid["day_num"].map(dict(enumerate(DAYS)))
    hourly = grid.merge(
        filtered[["day_num", "hour", "occupied"]],
        on=["day_num", "hour"],
        how="left",
    )
    hourly["occupied"] = hourly["occupied"].fillna(0)
    hourly["estimated_chargers"] = stations

    # Occupancy rate = chargers in use / total chargers. This is the real driver of
    # availability: a window is only "open" if chargers are actually free, accounting
    # for cars that plugged in earlier and are still occupying their spot.
    hourly["occupancy_rate"] = (hourly["occupied"] / stations).clip(0, 1)
    hourly["congestion_score"] = hourly["occupancy_rate"]
    hourly["availability_probability"] = (1 - hourly["occupancy_rate"]).clip(0.05, 0.95)

    hourly["congestion_level"] = pd.cut(
        hourly["congestion_score"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    hourly["hour_label"] = hourly["hour"].apply(format_hour)
    return hourly.sort_values(["day_num", "hour"])


def format_hour(hour):
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12
    h = 12 if h == 0 else h
    return f"{h}:00 {suffix}"


def find_best_window(day_df, start_hour, end_hour, window_hours=3):
    day_df = day_df[(day_df["hour"] >= start_hour) & (day_df["hour"] <= end_hour)].copy()
    if len(day_df) == 0:
        return None

    candidates = []
    latest_start = max(start_hour, end_hour - window_hours + 1)
    for h in range(start_hour, latest_start + 1):
        block = day_df[(day_df["hour"] >= h) & (day_df["hour"] < h + window_hours)]
        if len(block) == window_hours:
            candidates.append({
                "start_hour": h,
                "end_hour": h + window_hours,
                "avg_congestion": block["congestion_score"].mean(),
                "avg_probability": block["availability_probability"].mean(),
                "occupied": block["occupied"].mean(),
            })
    if not candidates:
        row = day_df.sort_values("congestion_score").iloc[0]
        return {
            "start_hour": int(row["hour"]),
            "end_hour": int(row["hour"] + 1),
            "avg_congestion": float(row["congestion_score"]),
            "avg_probability": float(row["availability_probability"]),
            "occupied": float(row["occupied"]),
        }
    return sorted(candidates, key=lambda x: x["avg_congestion"])[0]


def explain_recommendation(office, day, selected_hour, selected_row, window):
    selected_level = selected_row["congestion_level"]
    selected_prob = selected_row["availability_probability"]
    improvement = max(0, window["avg_probability"] - selected_prob)
    return (
        f"For {office}, historical patterns show {selected_level.lower()} congestion around "
        f"{format_hour(selected_hour)} on {day}, with an estimated {selected_prob:.0%} chance of finding an "
        f"open charger. The recommended window is {format_hour(window['start_hour'])}–"
        f"{format_hour(window['end_hour'])}, where estimated availability rises to {window['avg_probability']:.0%} "
        f"— about {improvement:.0%} better than your selected time. Estimates account for how long cars stay "
        "plugged in (Total Duration), not just when they arrive, and reflect historical patterns rather than "
        "live charger telemetry."
    )


OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
# Ollama is local — never route these calls through a corporate HTTP(S) proxy.
NO_PROXY = {"http": None, "https": None}


def ollama_models():
    """Return the list of locally available Ollama model names ([] if unreachable)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2, proxies=NO_PROXY)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def stream_ollama(model, system, messages):
    """Yield assistant text chunks from a local Ollama model (streaming chat)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
    }
    with requests.post(
        f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120, proxies=NO_PROXY
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def build_assistant_context(office, day, selected_hour, selected_row, window, chart_df):
    """System prompt: who the assistant is, plus a snapshot of the user's current view."""
    lines = [
        "You are the EV Charge Advisor assistant for NextEra Energy, helping employees "
        "plan when to charge their electric vehicles at the workplace.",
        "",
        "Use the live context below to answer questions about the current forecast and "
        "recommendation. You can also answer general questions on any topic. Be concise, "
        "friendly, and practical. When you use the data, explain it in plain language.",
        "",
        "Methodology: congestion is estimated from charger occupancy over time — each session "
        "is spread across every hour the car stays plugged in (Total Duration), so a car that "
        "plugs in at 7 AM and unplugs at 11 AM counts against 7-10 AM. Figures reflect historical "
        "patterns, not live telemetry, and cover only a sample of the site's chargers, so treat "
        "them as relative demand patterns and estimated availability.",
        "",
        "=== CURRENT VIEW ===",
        f"Office: {office}",
        f"Day: {day}",
        f"Planned arrival time: {format_hour(selected_hour)}",
        f"Forecasted congestion at arrival: {selected_row['congestion_level']}",
        f"Estimated availability at arrival: {selected_row['availability_probability']:.0%}",
    ]
    if window:
        lines.append(
            f"Recommended charging window: {format_hour(window['start_hour'])}–"
            f"{format_hour(window['end_hour'])} (estimated availability "
            f"{window['avg_probability']:.0%})"
        )
    lines.append("")
    lines.append("Hourly forecast for the selected day (hour | congestion | congestion % | est. availability):")
    for _, r in chart_df.iterrows():
        lines.append(
            f"- {r['hour_label']} | {r['congestion_level']} | "
            f"{r['congestion_pct']:.0f}% | {r['availability_probability']:.0%}"
        )
    return "\n".join(lines)


def render_chatbot(context):
    """Streaming chat UI backed by a local Ollama model."""
    st.markdown("**Ask the assistant**")
    st.caption("Ask about this forecast, the recommendation, or anything else.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    models = ollama_models()
    if not models:
        st.info(
            "The local AI assistant isn't available. Make sure Ollama is running "
            "(`ollama serve`) and a model is installed (`ollama pull llama3.2`)."
        )
        return

    model = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    # Fixed-height, scrollable chat area so the conversation stays inside the box
    # and scrolls internally instead of extending the page with each message.
    chat_box = st.container(height=320)
    with chat_box:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your charging forecast…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with chat_box:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages
                ]
                response = st.write_stream(stream_ollama(model, context, history))
            except Exception as e:
                response = f"Sorry, I couldn't reach the local AI model: {e}"
                st.error(response)

    st.session_state.chat_messages.append({"role": "assistant", "content": response})


df = load_data()
occ_table = build_occupancy_table(df)
offices = sorted(occ_table["office"].dropna().unique())

st.markdown(
    """
    <div class="nee-header">
      <div class="nee-logo">NextEra Energy<span>&#174;</span></div>
      <h1>EV Charge Advisor</h1>
      <p>AI-powered workplace charging demand forecast based on historical charging sessions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Plan your charge")
    office = st.selectbox("Office", offices)
    day = st.selectbox("Day", DAYS[:5], index=2)
    selected_hour = st.slider("What time would you arrive?", 5, 19, 9, format="%d:00")
    start_hour, end_hour = st.slider(
        "Hours you're flexible to charge between", 5, 20, (8, 17), format="%d:00"
    )
    window_hours = st.slider("Hours you need to charge", 1, 6, 2, format="%d hr")

hourly = build_hourly_table(occ_table, office)
day_df = hourly[hourly["day_name"] == day].copy()
selected_match = day_df[day_df["hour"] == selected_hour]
selected_row = selected_match.iloc[0] if len(selected_match) else day_df.iloc[0]
window = find_best_window(day_df, start_hour, end_hour, window_hours)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Forecasted congestion", selected_row["congestion_level"])
with col2:
    st.metric("Estimated availability", f"{selected_row['availability_probability']:.0%}")
with col3:
    if window:
        st.metric("Recommended window", f"{format_hour(window['start_hour'])}–{format_hour(window['end_hour'])}")
    else:
        st.metric("Recommended window", "No window found")

st.subheader(f"Forecasted congestion for {day}")
chart_df = day_df[(day_df["hour"] >= 5) & (day_df["hour"] <= 20)].copy().sort_values("hour")
chart_df["congestion_pct"] = (chart_df["congestion_score"] * 100).round(0)
hour_order = chart_df["hour_label"].tolist()  # keep x-axis in chronological order
fig = px.bar(
    chart_df,
    x="hour_label",
    y="congestion_pct",
    color="congestion_level",
    category_orders={"hour_label": hour_order, "congestion_level": ["Low", "Medium", "High"]},
    color_discrete_map=CONGESTION_COLORS,
    labels={"hour_label": "Hour", "congestion_pct": "Congestion (%)", "congestion_level": "Congestion"},
    title="Forecasted congestion by hour",
)
fig.update_yaxes(range=[0, 100])
fig.update_layout(
    font_color=NEXTERA_NAVY,
    title_font_color=NEXTERA_NAVY,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_font_color=NEXTERA_NAVY,
)
fig.update_xaxes(gridcolor="#e6edf2")
fig.update_yaxes(gridcolor="#e6edf2")
st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.subheader("AI explanation")
    if window:
        st.markdown(explain_recommendation(office, day, selected_hour, selected_row, window))
    st.divider()
    render_chatbot(build_assistant_context(office, day, selected_hour, selected_row, window, chart_df))

st.subheader("Daily details")
rec_hours = set(range(window["start_hour"], window["end_hour"])) if window else set()
show = chart_df[["hour_label", "hour", "congestion_level", "congestion_pct", "availability_probability"]].copy()
show["Recommended"] = show["hour"].map(lambda h: "✅" if h in rec_hours else "")
show["congestion_pct"] = show["congestion_pct"].map(lambda x: f"{x:.0f}%")
show["availability_probability"] = show["availability_probability"].map(lambda x: f"{x:.0%}")
show = show.drop(columns=["hour"]).rename(columns={
    "hour_label": "Hour",
    "congestion_level": "Congestion",
    "congestion_pct": "Congestion (%)",
    "availability_probability": "Est. availability",
})
st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("Data notes"):
    st.write(
        "This MVP estimates congestion from **charger occupancy over time**: each session is spread across "
        "every hour it stays plugged in (Total Duration), so a car that plugs in at 7 AM and unplugs at "
        "11 AM counts against 7, 8, 9 and 10 AM — not just 7 AM. Congestion for an hour reflects how heavily "
        "chargers are in use, including cars that arrived earlier and are still occupying spots."
    )
    st.write(
        "Total Duration (plug-in time) drives occupancy because a fully-charged car still blocks the spot. "
        "Charging Time (actual power draw) is typically shorter and is retained in the data for reference."
    )
    st.write(
        "Note: this export covers only a sample of the site's chargers, so the figures show **relative demand "
        "patterns and estimated availability**, not absolute charger counts or live telemetry."
    )
    st.write(f"Loaded {len(df):,} charging sessions.")
