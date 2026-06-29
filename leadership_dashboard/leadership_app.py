import os
import json

from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="EV Charge Advisor – Leadership", layout="wide")

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

def make_circular_avatar(path, size=256, padding=0.0):
    """Load an image, fit the whole thing (zoomed out) onto a square canvas,
    then mask it into a circle so Streamlit's avatar shows the full picture
    instead of a cropped/zoomed close-up."""
    img = Image.open(path).convert("RGBA")

    # Scale the full image to fit inside the circle, leaving a margin so it
    # isn't zoomed in. `padding` is the fraction of the canvas kept as margin.
    inner = int(size * (1 - padding))
    fitted = ImageOps.contain(img, (inner, inner))

    # Center the fitted image on a white square canvas. White matches the
    # icon's own background, so the leftover side bars blend in seamlessly
    # (a transparent canvas would turn black once the circular mask is applied).
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    canvas.paste(fitted, offset, fitted)

    # Apply a circular alpha mask so only the corners outside the circle are
    # transparent; everything inside stays the opaque white badge.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    canvas.putalpha(mask)
    return canvas


BOT_AVATAR = make_circular_avatar(Path(__file__).parent / "boticon.jpg")


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
    df["year_month"] = df["start_dt"].dt.to_period("M").astype(str)
    df["day_name"] = df["start_dt"].dt.day_name()
    df["hour"] = df["start_dt"].dt.hour
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
    table["occupancy_rate"] = (table["occupied"] / table["estimated_chargers"].clip(lower=1)).clip(0, 1)
    return table


def format_hour(hour):
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12
    h = 12 if h == 0 else h
    return f"{h}:00 {suffix}"


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


def build_assistant_context(df_f, occ_f, util_by_office, selected_offices,
                            co2_avoided_t=None, fleet_cost=None,
                            total_idle_hrs=None, pct_idle=None, mom_pct=None):
    """System prompt for the leadership AI assistant with fleet-level context."""
    total_sessions = len(df_f)
    total_energy = df_f["energy_kwh"].sum()
    unique_users = df_f["user_id"].nunique()
    total_chargers = df_f.groupby("office")["port_id"].nunique().sum()

    lines = [
        "You are the EV Charge Advisor assistant for NextEra Energy, supporting leadership and "
        "facilities teams with strategic insights into workplace EV charging infrastructure.",
        "",
        "Use the live fleet context below to answer questions about utilization, trends, "
        "capacity planning, and infrastructure decisions. You can also answer general questions. "
        "Be concise, data-driven, and practical. Translate numbers into actionable insights.",
        "",
        "Methodology: utilization is estimated from charger occupancy over time — each session "
        "is spread across every hour the car stays plugged in (Total Duration). Figures reflect "
        "historical patterns, not live telemetry, and cover only a sample of each site's chargers.",
        "",
        "=== CURRENT FLEET SNAPSHOT ===",
        f"Offices in view: {', '.join(selected_offices) if selected_offices else 'All'}",
        f"Total charging sessions: {total_sessions:,}",
        f"Total energy delivered: {total_energy:,.0f} kWh",
        f"Unique EV drivers: {unique_users:,}",
        f"Estimated charging ports: {int(total_chargers):,}",
    ]
    if co2_avoided_t is not None:
        lines.append(f"Estimated CO₂ avoided: {co2_avoided_t:,.1f} tonnes")
    if fleet_cost is not None:
        lines.append(f"Estimated fleet charging cost: ${fleet_cost:,.0f}")
    if total_idle_hrs is not None:
        lines.append(f"Total idle charger-hours (vehicles done but still plugged in): {total_idle_hrs:,.0f} hrs ({pct_idle:.1f}% of plug-in time)")
    if mom_pct is not None:
        lines.append(f"Fleet session growth trend: {mom_pct:+.1f}% month-over-month (linear)")
    lines.append("")
    lines.append("Utilization by office (avg business-hours utilization rate | status):")
    for _, row in util_by_office.iterrows():
        lines.append(
            f"- {row['office']}: {row['avg_utilization_pct']:.1f}% ({row['status']})"
        )
    return "\n".join(lines)


def render_chatbot(context):
    """Streaming chat UI backed by a local Ollama model."""
    st.markdown("**Ask the assistant**")
    st.caption("Ask about fleet utilization, infrastructure planning, or anything else.")

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

    chat_box = st.container(height=320)
    with chat_box:
        for msg in st.session_state.chat_messages:
            avatar = BOT_AVATAR if msg["role"] == "assistant" else "user"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask about fleet data, utilization, or infrastructure…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with chat_box:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
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

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="nee-header">
      <div class="nee-logo">NextEra Energy<span>&#174;</span></div>
      <h1>EV Charge Advisor – Leadership Dashboard</h1>
      <p>Strategic insights into workplace EV charging utilization and infrastructure planning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    selected_offices = st.multiselect("Offices", offices, default=offices)
    months_available = sorted(df["year_month"].unique())
    selected_months = st.multiselect("Months", months_available, default=months_available)
    st.divider()
    st.header("Assumptions")
    electricity_cost = st.number_input("Electricity cost ($/kWh)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    co2_per_kwh = st.number_input("CO₂ saved per kWh (kg)", min_value=0.1, max_value=2.0, value=0.7, step=0.05,
                                   help="Average CO₂ per kWh avoided vs. a gasoline vehicle.")

# Apply filters
df_f = df[df["office"].isin(selected_offices) & df["year_month"].isin(selected_months)]
occ_f = occ_table[occ_table["office"].isin(selected_offices)]

# ── Month-over-month delta helper ──────────────────────────────────────────────
def _mom_delta(df_full, offices_sel, col, agg="sum"):
    """Return (current_val, delta_vs_prev_month) for the most recent two months."""
    sub = df_full[df_full["office"].isin(offices_sel)] if offices_sel else df_full
    months = sorted(sub["year_month"].unique())
    if len(months) < 2:
        return None, None
    cur_m, prev_m = months[-1], months[-2]
    cur = sub[sub["year_month"] == cur_m][col]
    prev = sub[sub["year_month"] == prev_m][col]
    if agg == "sum":
        cur_val, prev_val = cur.sum(), prev.sum()
    elif agg == "nunique":
        cur_val, prev_val = cur.nunique(), prev.nunique()
    else:  # count
        cur_val, prev_val = len(cur), len(prev)
    delta = cur_val - prev_val
    return cur_val, delta

# ── Fleet-level KPIs ───────────────────────────────────────────────────────────
st.subheader("Fleet overview")
total_sessions = len(df_f)
total_energy = df_f["energy_kwh"].sum()
unique_users = df_f["user_id"].nunique()
total_chargers = df_f.groupby("office")["port_id"].nunique().sum()
avg_session_kwh = total_energy / total_sessions if total_sessions else 0
avg_dur_hrs = (
    df_f["total_duration"].dropna().apply(lambda x: x.total_seconds() / 3600).mean()
    if len(df_f) > 0 else 0
)

_, delta_sessions = _mom_delta(df, selected_offices, "user_id", agg="count")
_, delta_energy   = _mom_delta(df, selected_offices, "energy_kwh", agg="sum")
_, delta_users    = _mom_delta(df, selected_offices, "user_id", agg="nunique")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total sessions",        f"{total_sessions:,}",       delta=f"{delta_sessions:+,.0f} vs prev mo" if delta_sessions is not None else None)
k2.metric("Total energy delivered", f"{total_energy:,.0f} kWh", delta=f"{delta_energy:+,.0f} kWh vs prev mo" if delta_energy is not None else None)
k3.metric("Unique users",           f"{unique_users:,}",        delta=f"{delta_users:+,.0f} vs prev mo" if delta_users is not None else None)
k4.metric("Charging ports (est.)",  f"{int(total_chargers):,}")
k5.metric("Avg energy / session",   f"{avg_session_kwh:.1f} kWh")
k6.metric("Avg session duration",   f"{avg_dur_hrs:.1f} hrs")

st.divider()

# ── Sustainability impact ──────────────────────────────────────────────────────
st.subheader("Sustainability impact")
st.caption("Estimates based on sidebar assumptions. CO₂ savings vs. average gasoline vehicle; cost = fleet total charging spend.")

co2_avoided_kg   = total_energy * co2_per_kwh
co2_avoided_t    = co2_avoided_kg / 1000
tree_years        = co2_avoided_kg / 21          # ~21 kg CO₂ absorbed per tree per year
gasoline_litres   = total_energy / 8.9           # ~8.9 kWh per litre of gasoline equivalent
fleet_cost        = total_energy * electricity_cost

s1, s2, s3, s4 = st.columns(4)
s1.metric("CO₂ avoided",         f"{co2_avoided_t:,.1f} tonnes")
s2.metric("Tree-years equivalent", f"{tree_years:,.0f} tree-yrs")
s3.metric("Gasoline offset",      f"{gasoline_litres:,.0f} L")
s4.metric("Est. fleet charge cost", f"${fleet_cost:,.0f}")

st.divider()

# ── Sessions & energy by office ────────────────────────────────────────────────
st.subheader("Utilization by office")
col_a, col_b = st.columns(2)

office_sessions = (
    df_f.groupby("office", as_index=False)
    .agg(sessions=("user_id", "count"), energy_kwh=("energy_kwh", "sum"))
    .sort_values("sessions", ascending=False)
)

def _branded_bar(df_in, x, y, title, label_y, color):
    fig = px.bar(
        df_in, x=x, y=y,
        labels={x: "Office", y: label_y},
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        font_color=NEXTERA_NAVY,
        title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_tickangle=-30,
    )
    fig.update_xaxes(gridcolor="#e6edf2")
    fig.update_yaxes(gridcolor="#e6edf2")
    return fig

with col_a:
    st.plotly_chart(
        _branded_bar(office_sessions, "office", "sessions",
                     "Total charging sessions by office", "Sessions", NEXTERA_BLUE),
        use_container_width=True,
    )

with col_b:
    st.plotly_chart(
        _branded_bar(office_sessions, "office", "energy_kwh",
                     "Total energy delivered by office", "Energy (kWh)", NEXTERA_GREEN),
        use_container_width=True,
    )

st.divider()

# ── Monthly trend ──────────────────────────────────────────────────────────────
st.subheader("Monthly demand trend")
monthly = (
    df_f.groupby(["year_month", "office"], as_index=False)
    .agg(sessions=("user_id", "count"), energy_kwh=("energy_kwh", "sum"))
    .sort_values("year_month")
)

def _branded_line(df_in, x, y, title, label_y):
    fig = px.line(
        df_in, x=x, y=y, color="office", markers=True,
        labels={x: "Month", y: label_y, "office": "Office"},
        title=title,
        color_discrete_sequence=[NEXTERA_BLUE, NEXTERA_GREEN, NEXTERA_BLUE_LIGHT, NEXTERA_RED],
    )
    fig.update_layout(
        font_color=NEXTERA_NAVY,
        title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_tickangle=-30,
    )
    fig.update_xaxes(gridcolor="#e6edf2")
    fig.update_yaxes(gridcolor="#e6edf2")
    return fig

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(
        _branded_line(monthly, "year_month", "sessions",
                      "Monthly session volume by office", "Sessions"),
        use_container_width=True,
    )
with col_d:
    st.plotly_chart(
        _branded_line(monthly, "year_month", "energy_kwh",
                      "Monthly energy delivered by office", "Energy (kWh)"),
        use_container_width=True,
    )

# Fleet-wide growth + 3-month linear projection
fleet_monthly = (
    df_f.groupby("year_month", as_index=False)
    .agg(sessions=("user_id", "count"))
    .sort_values("year_month")
)
if len(fleet_monthly) >= 3:
    x_idx = np.arange(len(fleet_monthly))
    coeffs = np.polyfit(x_idx, fleet_monthly["sessions"].values, 1)
    slope, intercept = coeffs
    mom_pct = (slope / fleet_monthly["sessions"].mean() * 100) if fleet_monthly["sessions"].mean() else 0

    proj_months = pd.period_range(
        start=pd.Period(fleet_monthly["year_month"].iloc[-1], "M") + 1, periods=3, freq="M"
    ).astype(str).tolist()
    proj_x = np.arange(len(fleet_monthly), len(fleet_monthly) + 3)
    proj_vals = np.polyval(coeffs, proj_x).clip(0).tolist()

    hist_df = fleet_monthly.copy()
    hist_df["type"] = "Historical"
    proj_df = pd.DataFrame({"year_month": proj_months, "sessions": proj_vals, "type": "Projected"})
    growth_df = pd.concat([hist_df, proj_df], ignore_index=True)

    fig_growth = px.bar(
        growth_df, x="year_month", y="sessions", color="type",
        color_discrete_map={"Historical": NEXTERA_BLUE, "Projected": NEXTERA_BLUE_LIGHT},
        labels={"year_month": "Month", "sessions": "Sessions", "type": ""},
        title=f"Fleet-wide session volume + 3-month projection  "
              f"({'growing' if slope > 0 else 'declining'} at {abs(mom_pct):.1f}% MoM)",
    )
    # Overlay trendline
    trend_x = list(fleet_monthly["year_month"]) + proj_months
    trend_y = np.polyval(coeffs, np.arange(len(trend_x))).clip(0).tolist()
    fig_growth.add_scatter(
        x=trend_x, y=trend_y, mode="lines",
        line=dict(color=NEXTERA_RED, width=2, dash="dot"),
        name="Trend",
    )
    fig_growth.update_layout(
        font_color=NEXTERA_NAVY, title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30,
    )
    fig_growth.update_xaxes(gridcolor="#e6edf2")
    fig_growth.update_yaxes(gridcolor="#e6edf2")
    st.plotly_chart(fig_growth, use_container_width=True)

st.divider()

# ── Average charger utilization rate by office ─────────────────────────────────
st.subheader("Average charger utilization rate by office")
st.caption(
    "Utilization rate = average fraction of chargers in simultaneous use during business hours (6 AM – 8 PM). "
    "A rate above 70 % typically signals a need for additional infrastructure."
)

biz_hours = occ_f[(occ_f["hour"] >= 6) & (occ_f["hour"] <= 20)]
util_by_office = (
    biz_hours.groupby("office", as_index=False)
    .agg(avg_utilization=("occupancy_rate", "mean"))
    .sort_values("avg_utilization", ascending=False)
)
util_by_office["avg_utilization_pct"] = (util_by_office["avg_utilization"] * 100).round(1)
util_by_office["status"] = util_by_office["avg_utilization"].apply(
    lambda x: "Critical" if x >= 0.70 else ("Moderate" if x >= 0.45 else "Healthy")
)

status_colors = {"Critical": NEXTERA_RED, "Moderate": "#e07b00", "Healthy": NEXTERA_GREEN}
fig_util = px.bar(
    util_by_office,
    x="office",
    y="avg_utilization_pct",
    color="status",
    color_discrete_map=status_colors,
    labels={"office": "Office", "avg_utilization_pct": "Avg utilization (%)", "status": "Status"},
    title="Average charger utilization rate (business hours)",
)
fig_util.update_yaxes(range=[0, 100])
fig_util.update_layout(
    font_color=NEXTERA_NAVY,
    title_font_color=NEXTERA_NAVY,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_tickangle=-30,
)
fig_util.update_xaxes(gridcolor="#e6edf2")
fig_util.update_yaxes(gridcolor="#e6edf2")
st.plotly_chart(fig_util, use_container_width=True)

st.divider()

# ── Day-of-week demand heatmap ──────────────────────────────────────────────────
st.subheader("Demand heatmap – day of week vs. hour")
heatmap_offices = selected_offices if selected_offices else offices
selected_office_heat = st.selectbox("Select office for heatmap", heatmap_offices)

heat_data = occ_f[occ_f["office"] == selected_office_heat].copy()
heat_pivot = heat_data.pivot_table(
    index="day_name", columns="hour", values="occupancy_rate", aggfunc="mean"
)
heat_pivot = heat_pivot.reindex(columns=range(24), fill_value=0)
heat_pivot = heat_pivot.reindex([d for d in DAYS if d in heat_pivot.index])
hour_labels = [format_hour(h) for h in range(24)]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heat_pivot.values,
        x=hour_labels,
        y=heat_pivot.index.tolist(),
        colorscale=[[0, "#e8f4f8"], [0.5, NEXTERA_BLUE], [1, NEXTERA_NAVY]],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Utilization rate", tickformat=".0%"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Utilization: %{z:.1%}<extra></extra>",
    )
)
fig_heat.update_layout(
    title=f"Charger utilization heatmap – {selected_office_heat}",
    xaxis_title="Hour of day",
    yaxis_title="Day of week",
    height=400,
    font_color=NEXTERA_NAVY,
    title_font_color=NEXTERA_NAVY,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Peak hours by office ────────────────────────────────────────────────────────
st.subheader("Peak demand hours by office")
peak_hours = (
    occ_f[(occ_f["hour"] >= 6) & (occ_f["hour"] <= 20)]
    .groupby(["office", "hour"], as_index=False)
    .agg(avg_occupancy=("occupancy_rate", "mean"))
)
peak_hours["hour_label"] = peak_hours["hour"].apply(format_hour)
hour_order = [format_hour(h) for h in range(6, 21)]

fig_peak = px.bar(
    peak_hours,
    x="hour_label",
    y="avg_occupancy",
    color="office",
    barmode="group",
    category_orders={"hour_label": hour_order},
    labels={"hour_label": "Hour", "avg_occupancy": "Avg utilization rate", "office": "Office"},
    title="Average charger utilization by hour across offices",
    color_discrete_sequence=[NEXTERA_BLUE, NEXTERA_GREEN, NEXTERA_BLUE_LIGHT, NEXTERA_RED],
)
fig_peak.update_yaxes(tickformat=".0%", range=[0, 1], gridcolor="#e6edf2")
fig_peak.update_xaxes(gridcolor="#e6edf2")
fig_peak.update_layout(
    font_color=NEXTERA_NAVY,
    title_font_color=NEXTERA_NAVY,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_peak, use_container_width=True)

st.divider()

# ── Charging efficiency – idle time analysis ───────────────────────────────────
st.subheader("Charging efficiency – idle time analysis")
st.caption(
    "Idle time = time the car stays plugged in after charging is complete (Total Duration − Charging Time). "
    "Cars that hoard a charger after finishing block other drivers. Reducing idle time is the fastest "
    "way to improve availability without adding hardware."
)

idle_df = df_f.copy()
idle_df["idle_hrs"] = (
    (idle_df["total_duration"] - idle_df["charging_time"])
    .apply(lambda x: x.total_seconds() / 3600 if pd.notna(x) and x.total_seconds() > 0 else 0)
)
idle_df["charging_hrs"] = idle_df["charging_time"].apply(
    lambda x: x.total_seconds() / 3600 if pd.notna(x) else 0
)

idle_by_office = (
    idle_df.groupby("office", as_index=False)
    .agg(
        avg_idle_hrs=("idle_hrs", "mean"),
        total_idle_hrs=("idle_hrs", "sum"),
        avg_charging_hrs=("charging_hrs", "mean"),
    )
    .sort_values("avg_idle_hrs", ascending=False)
)

col_idle_a, col_idle_b = st.columns(2)
with col_idle_a:
    fig_idle_avg = px.bar(
        idle_by_office, x="office", y="avg_idle_hrs",
        labels={"office": "Office", "avg_idle_hrs": "Avg idle time (hrs)"},
        title="Average idle time per session by office",
        color_discrete_sequence=[NEXTERA_RED],
    )
    fig_idle_avg.update_layout(
        font_color=NEXTERA_NAVY, title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30,
    )
    fig_idle_avg.update_xaxes(gridcolor="#e6edf2")
    fig_idle_avg.update_yaxes(gridcolor="#e6edf2")
    st.plotly_chart(fig_idle_avg, use_container_width=True)

with col_idle_b:
    fig_idle_total = px.bar(
        idle_by_office, x="office", y="total_idle_hrs",
        labels={"office": "Office", "total_idle_hrs": "Total idle charger-hours"},
        title="Total charger-hours wasted by idle vehicles",
        color_discrete_sequence=["#e07b00"],
    )
    fig_idle_total.update_layout(
        font_color=NEXTERA_NAVY, title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30,
    )
    fig_idle_total.update_xaxes(gridcolor="#e6edf2")
    fig_idle_total.update_yaxes(gridcolor="#e6edf2")
    st.plotly_chart(fig_idle_total, use_container_width=True)

total_idle_hrs = idle_df["idle_hrs"].sum()
pct_idle = (total_idle_hrs / idle_df["total_duration"].dropna().apply(
    lambda x: x.total_seconds() / 3600).sum() * 100) if len(idle_df) > 0 else 0
st.info(
    f"Across all selected offices, **{total_idle_hrs:,.0f} charger-hours** were spent idle "
    f"({pct_idle:.1f}% of total plug-in time). Notifying drivers when charging is complete "
    "could reclaim a significant fraction of that capacity."
)

st.divider()

# ── Infrastructure planning table ──────────────────────────────────────────────
st.subheader("Infrastructure planning summary")
st.caption("Use this table to identify offices approaching capacity and prioritize charger expansion.")

infra = df_f.groupby("office", as_index=False).agg(
    total_sessions=("user_id", "count"),
    unique_users=("user_id", "nunique"),
    total_energy_kwh=("energy_kwh", "sum"),
    estimated_chargers=("port_id", "nunique"),
)
util_lookup = util_by_office.set_index("office")[["avg_utilization_pct", "status"]]
infra = infra.merge(util_lookup, on="office", how="left")
infra["sessions_per_charger"] = (infra["total_sessions"] / infra["estimated_chargers"].clip(lower=1)).round(1)
infra = infra.sort_values("avg_utilization_pct", ascending=False)
infra["total_energy_kwh"] = infra["total_energy_kwh"].round(0)
infra = infra.rename(columns={
    "office": "Office",
    "total_sessions": "Sessions",
    "unique_users": "Unique users",
    "total_energy_kwh": "Energy (kWh)",
    "estimated_chargers": "Charger ports",
    "avg_utilization_pct": "Avg utilization (%)",
    "status": "Status",
    "sessions_per_charger": "Sessions / charger",
})
st.dataframe(infra, use_container_width=True, hide_index=True)

st.divider()

# ── Top power users ────────────────────────────────────────────────────────────
st.subheader("Top power users")
st.caption("Anonymized by last 4 digits of User ID. Useful for identifying candidates for at-home charging subsidies or fleet EV policy reviews.")

user_stats = (
    df_f.groupby("user_id", as_index=False)
    .agg(sessions=("user_id", "count"), energy_kwh=("energy_kwh", "sum"))
    .assign(label=lambda d: d["user_id"].astype(str).str[-4:].apply(lambda x: f"User …{x}"))
)

col_top_a, col_top_b = st.columns(2)
with col_top_a:
    top_energy = user_stats.nlargest(10, "energy_kwh")
    fig_top_e = px.bar(
        top_energy, x="energy_kwh", y="label", orientation="h",
        labels={"energy_kwh": "Total energy (kWh)", "label": ""},
        title="Top 10 users by energy consumed",
        color_discrete_sequence=[NEXTERA_BLUE],
    )
    fig_top_e.update_layout(
        font_color=NEXTERA_NAVY, title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(categoryorder="total ascending"),
    )
    fig_top_e.update_xaxes(gridcolor="#e6edf2")
    fig_top_e.update_yaxes(gridcolor="#e6edf2")
    st.plotly_chart(fig_top_e, use_container_width=True)

with col_top_b:
    top_sessions = user_stats.nlargest(10, "sessions")
    fig_top_s = px.bar(
        top_sessions, x="sessions", y="label", orientation="h",
        labels={"sessions": "Total sessions", "label": ""},
        title="Top 10 users by session count",
        color_discrete_sequence=[NEXTERA_GREEN],
    )
    fig_top_s.update_layout(
        font_color=NEXTERA_NAVY, title_font_color=NEXTERA_NAVY,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(categoryorder="total ascending"),
    )
    fig_top_s.update_xaxes(gridcolor="#e6edf2")
    fig_top_s.update_yaxes(gridcolor="#e6edf2")
    st.plotly_chart(fig_top_s, use_container_width=True)

st.divider()

# ── AI Assistant ───────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("AI Assistant")
    _ai_mom_pct = locals().get("mom_pct")
    render_chatbot(build_assistant_context(
        df_f, occ_f, util_by_office, selected_offices,
        co2_avoided_t=co2_avoided_t,
        fleet_cost=fleet_cost,
        total_idle_hrs=total_idle_hrs,
        pct_idle=pct_idle,
        mom_pct=_ai_mom_pct,
    ))

with st.expander("Data notes"):
    st.write(
        "Utilization is estimated from **charger occupancy over time**: each session is spread across "
        "every hour the car stays plugged in (Total Duration), so a car that plugs in at 7 AM and unplugs at "
        "11 AM counts against 7, 8, 9 and 10 AM. Figures reflect historical patterns, not live telemetry."
    )
    st.write(
        "Note: this export covers only a sample of each site's chargers, so figures show **relative demand "
        "patterns and estimated utilization**, not absolute charger counts."
    )
    st.write(f"Loaded {len(df):,} charging sessions.")
