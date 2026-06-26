import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="EV Charge Advisor – Leadership", layout="wide", page_icon="⚡")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #eef2ff 0%, #e8edff 100%);
    border: 1px solid #c5cfee;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(37,99,235,0.07);
}
div[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.04em; }
</style>
""", unsafe_allow_html=True)

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
    df["port_id"] = df["Station Name"].astype(str) + " / port " + df["Port Number"].astype(str)
    df["user_id"] = df["User ID"]
    df["energy_kwh"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce").fillna(0)
    df["total_duration"] = df["Total Duration (hh:mm:ss)"].apply(parse_hms)
    df["charging_time"] = df["Charging Time (hh:mm:ss)"].apply(parse_hms)

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
    dur_secs = df["total_duration"].apply(lambda x: x.total_seconds() if pd.notna(x) else 0)
    chg_secs = df["charging_time"].apply(lambda x: x.total_seconds() if pd.notna(x) else 0)
    df["idle_hrs"] = ((dur_secs - chg_secs).clip(lower=0) / 3600)
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
        while cur < end and guard < 240:
            hour_start = cur
            hour_end = cur + pd.Timedelta(hours=1)
            overlap = (min(end, hour_end) - max(start, hour_start)).total_seconds()
            if overlap > 0:
                records.append((
                    r.office,
                    cur.normalize(),
                    cur.dayofweek,
                    cur.hour,
                    overlap / 3600.0,
                ))
            cur = hour_end
            guard += 1

    occ = pd.DataFrame(records, columns=["office", "date", "day_num", "hour", "occ_frac"])

    per_date = occ.groupby(["office", "date", "day_num", "hour"], as_index=False).agg(
        occupied=("occ_frac", "sum"),
    )
    table = per_date.groupby(["office", "day_num", "hour"], as_index=False).agg(
        occupied=("occupied", "mean"),
    )
    table["day_name"] = table["day_num"].map(dict(enumerate(DAYS)))

    stations = df.groupby("office")["port_id"].nunique().rename("estimated_chargers")
    table = table.merge(stations, on="office", how="left")
    table["occupancy_rate"] = (table["occupied"] / table["estimated_chargers"].clip(lower=1)).clip(0, 1)
    return table


def format_hour(hour):
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12
    h = 12 if h == 0 else h
    return f"{h}:00 {suffix}"


# ── Load data ──────────────────────────────────────────────────────────────────
df = load_data()
occ_table = build_occupancy_table(df)
offices = sorted(occ_table["office"].dropna().unique())

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("EV Charge Advisor – Leadership Dashboard")
st.caption("Strategic insights into workplace EV charging utilization and infrastructure planning")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    selected_offices = st.multiselect("Offices", offices, default=offices)
    months_available = sorted(df["year_month"].unique())
    selected_months = st.multiselect("Months", months_available, default=months_available)
    st.divider()
    st.header("Assumptions")
    cost_per_kwh = st.slider("Electricity cost ($/kWh)", 0.05, 0.50, 0.15, 0.01,
                             help="Used to estimate total fleet charging cost")
    co2_kg_per_kwh = st.slider("CO₂ saved per kWh (kg)", 0.3, 1.5, 0.7, 0.05,
                               help="Estimated kg CO₂ avoided vs average ICE vehicle per kWh delivered")

# Apply filters
df_f = df[df["office"].isin(selected_offices) & df["year_month"].isin(selected_months)]
occ_f = occ_table[occ_table["office"].isin(selected_offices)]

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
fleet_cost = total_energy * cost_per_kwh
co2_saved_kg = total_energy * co2_kg_per_kwh

# MoM delta
sorted_months_all = sorted(df_f["year_month"].unique())
if len(sorted_months_all) >= 2:
    m_curr, m_prev = sorted_months_all[-1], sorted_months_all[-2]
    sessions_delta = len(df_f[df_f["year_month"] == m_curr]) - len(df_f[df_f["year_month"] == m_prev])
    energy_delta = (
        df_f[df_f["year_month"] == m_curr]["energy_kwh"].sum()
        - df_f[df_f["year_month"] == m_prev]["energy_kwh"].sum()
    )
else:
    sessions_delta = energy_delta = None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total sessions", f"{total_sessions:,}",
          delta=f"{sessions_delta:+,} vs prev month" if sessions_delta is not None else None)
k2.metric("Total energy delivered", f"{total_energy:,.0f} kWh",
          delta=f"{energy_delta:+,.0f} kWh" if energy_delta is not None else None)
k3.metric("Unique users", f"{unique_users:,}")
k4.metric("Charging ports (est.)", f"{int(total_chargers):,}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Avg energy / session", f"{avg_session_kwh:.1f} kWh")
k6.metric("Avg session duration", f"{avg_dur_hrs:.1f} hrs")
k7.metric("Estimated fleet cost", f"${fleet_cost:,.0f}", help=f"Based on ${cost_per_kwh}/kWh")
k8.metric("CO₂ avoided (est.)", f"{co2_saved_kg / 1000:.1f} t", help=f"At {co2_kg_per_kwh} kg CO₂/kWh vs ICE")

st.divider()

# ── Sessions & energy by office ────────────────────────────────────────────────
st.subheader("Utilization by office")
col_a, col_b = st.columns(2)

office_sessions = (
    df_f.groupby("office", as_index=False)
    .agg(sessions=("user_id", "count"), energy_kwh=("energy_kwh", "sum"))
    .sort_values("sessions", ascending=False)
)

with col_a:
    fig_sessions = px.bar(
        office_sessions,
        x="office",
        y="sessions",
        labels={"office": "Office", "sessions": "Sessions"},
        title="Total charging sessions by office",
        color="sessions",
        color_continuous_scale="Blues",
    )
    fig_sessions.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_sessions, use_container_width=True)

with col_b:
    fig_energy = px.bar(
        office_sessions,
        x="office",
        y="energy_kwh",
        labels={"office": "Office", "energy_kwh": "Energy (kWh)"},
        title="Total energy delivered by office",
        color="energy_kwh",
        color_continuous_scale="Greens",
    )
    fig_energy.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_energy, use_container_width=True)

st.divider()

# ── Monthly trend ──────────────────────────────────────────────────────────────
st.subheader("Monthly demand trend")
monthly = (
    df_f.groupby(["year_month", "office"], as_index=False)
    .agg(sessions=("user_id", "count"), energy_kwh=("energy_kwh", "sum"))
    .sort_values("year_month")
)

col_c, col_d = st.columns(2)
with col_c:
    fig_trend = px.line(
        monthly,
        x="year_month",
        y="sessions",
        color="office",
        markers=True,
        labels={"year_month": "Month", "sessions": "Sessions", "office": "Office"},
        title="Monthly session volume by office",
    )
    fig_trend.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_d:
    fig_energy_trend = px.line(
        monthly,
        x="year_month",
        y="energy_kwh",
        color="office",
        markers=True,
        labels={"year_month": "Month", "energy_kwh": "Energy (kWh)", "office": "Office"},
        title="Monthly energy delivered by office",
    )
    fig_energy_trend.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_energy_trend, use_container_width=True)

# ── Growth trajectory & projection ────────────────────────────────────────────────
monthly_total = (
    df_f.groupby("year_month", as_index=False)
    .agg(all_sessions=("user_id", "count"))
    .sort_values("year_month")
)
if len(monthly_total) >= 3:
    x_vals = list(range(len(monthly_total)))
    y_vals = monthly_total["all_sessions"].tolist()
    coeffs = np.polyfit(x_vals, y_vals, 1)
    slope = coeffs[0]
    last_period = pd.Period(monthly_total["year_month"].iloc[-1], freq="M")
    proj_labels = [(last_period + i).strftime("%Y-%m") for i in range(1, 4)]
    proj_y = [max(0, int(coeffs[1] + slope * (len(monthly_total) + i - 1))) for i in range(1, 4)]

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(
        x=monthly_total["year_month"], y=monthly_total["all_sessions"],
        name="Actual", marker_color="#2563eb",
    ))
    fig_proj.add_trace(go.Bar(
        x=proj_labels, y=proj_y,
        name="Projected", marker_color="rgba(251,146,60,0.5)",
        marker_line=dict(color="#f97316", width=2),
    ))
    trend_x = monthly_total["year_month"].tolist() + proj_labels
    trend_y = [coeffs[1] + slope * i for i in range(len(trend_x))]
    fig_proj.add_trace(go.Scatter(
        x=trend_x, y=[max(0, v) for v in trend_y],
        mode="lines", name="Trend line",
        line=dict(color="#f97316", dash="dot", width=2),
    ))
    fig_proj.update_layout(
        title="Fleet-wide session growth & 3-month projection",
        xaxis_title="Month", yaxis_title="Total sessions",
        xaxis_tickangle=-30, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    mom_pct = (slope / (sum(y_vals) / len(y_vals)) * 100) if y_vals else 0
    direction = "📈 growing" if slope > 0 else "📉 declining"
    st.info(
        f"**Demand trend:** Fleet charging demand is {direction} at approximately "
        f"**{abs(slope):.0f} sessions/month ({abs(mom_pct):.1f}% MoM)**. "
        f"At this rate, the fleet is projected to reach **{proj_y[-1]:,} sessions/month** in 3 months."
    )

st.divider()

# ── Sustainability impact ───────────────────────────────────────────────────────────────
st.subheader("🌱 Sustainability impact")
st.caption("Estimates based on adjustable assumptions in the sidebar.")
trees_equiv = co2_saved_kg / 21
gas_liters_equiv = total_energy * 0.35
s1, s2, s3, s4 = st.columns(4)
s1.metric("CO₂ avoided", f"{co2_saved_kg / 1000:.1f} tonnes",
          help=f"{co2_kg_per_kwh} kg CO₂/kWh assumption")
s2.metric("Tree-years equivalent", f"{trees_equiv:,.0f}",
          help="Based on avg tree absorbing 21 kg CO₂/year")
s3.metric("Gasoline offset (est.)", f"{gas_liters_equiv:,.0f} L",
          help="~0.35 L gasoline equivalent replaced per kWh delivered")
s4.metric("Estimated fleet cost", f"${fleet_cost:,.0f}",
          help=f"${cost_per_kwh}/kWh electricity rate")

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

color_map = {"Critical": "#d62728", "Moderate": "#ff7f0e", "Healthy": "#2ca02c"}
fig_util = px.bar(
    util_by_office,
    x="office",
    y="avg_utilization_pct",
    color="status",
    color_discrete_map=color_map,
    labels={"office": "Office", "avg_utilization_pct": "Avg utilization (%)", "status": "Status"},
    title="Average charger utilization rate (business hours)",
)
fig_util.update_yaxes(range=[0, 100])
fig_util.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_util, use_container_width=True)

st.divider()

# ── Charging efficiency – idle time ─────────────────────────────────────────────────
st.subheader("⏱️ Charging efficiency – idle time analysis")
st.caption(
    "Idle time = total plug-in duration minus active charging time. "
    "High idle time means vehicles occupy chargers after charging is complete, blocking others."
)
idle_by_office = (
    df_f[df_f["idle_hrs"] > 0]
    .groupby("office", as_index=False)
    .agg(
        avg_idle_hrs=("idle_hrs", "mean"),
        total_idle_hrs=("idle_hrs", "sum"),
        sessions_with_idle=("idle_hrs", "count"),
    )
    .sort_values("avg_idle_hrs", ascending=False)
)
if len(idle_by_office) > 0:
    col_eff1, col_eff2 = st.columns(2)
    with col_eff1:
        fig_idle = px.bar(
            idle_by_office, x="office", y="avg_idle_hrs",
            color="avg_idle_hrs", color_continuous_scale="OrRd",
            labels={"office": "Office", "avg_idle_hrs": "Avg idle time (hrs)"},
            title="Average idle time per session by office",
            text=idle_by_office["avg_idle_hrs"].apply(lambda x: f"{x:.1f}h"),
        )
        fig_idle.update_traces(textposition="outside")
        fig_idle.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_idle, use_container_width=True)
    with col_eff2:
        fig_idle_total = px.bar(
            idle_by_office, x="office", y="total_idle_hrs",
            color="total_idle_hrs", color_continuous_scale="Reds",
            labels={"office": "Office", "total_idle_hrs": "Total idle hours"},
            title="Total charger-hours wasted to idle occupancy",
            text=idle_by_office["total_idle_hrs"].apply(lambda x: f"{x:,.0f}h"),
        )
        fig_idle_total.update_traces(textposition="outside")
        fig_idle_total.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_idle_total, use_container_width=True)
else:
    st.info("No idle time data available for the selected filters.")

st.divider()

# ── Day-of-week demand heatmap ──────────────────────────────────────────────────
st.subheader("Demand heatmap – day of week vs. hour")
selected_office_heat = st.selectbox("Select office for heatmap", selected_offices if selected_offices else offices)

heat_data = occ_f[occ_f["office"] == selected_office_heat].copy()
heat_pivot = heat_data.pivot_table(
    index="day_name", columns="hour", values="occupancy_rate", aggfunc="mean"
)
# Ensure all 24 hours present and days in order
heat_pivot = heat_pivot.reindex(columns=range(24), fill_value=0)
heat_pivot = heat_pivot.reindex([d for d in DAYS if d in heat_pivot.index])

hour_labels = [format_hour(h) for h in range(24)]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heat_pivot.values,
        x=hour_labels,
        y=heat_pivot.index.tolist(),
        colorscale="RdYlGn_r",
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
)
fig_peak.update_yaxes(tickformat=".0%", range=[0, 1])
fig_peak.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_peak, use_container_width=True)

st.divider()

# ── Top power users ─────────────────────────────────────────────────────────────────
st.subheader("👤 Top power users")
st.caption("Anonymized employee charging profiles — User IDs are truncated for privacy.")

top_by_energy = (
    df_f.groupby("user_id", as_index=False)
    .agg(sessions=("user_id", "count"), total_kwh=("energy_kwh", "sum"))
    .sort_values("total_kwh", ascending=False)
    .head(10)
)
top_by_energy["user_label"] = top_by_energy["user_id"].astype(str).apply(lambda x: f"User …{x[-4:]}")
top_by_energy["total_kwh"] = top_by_energy["total_kwh"].round(1)

top_by_sessions = (
    df_f.groupby("user_id", as_index=False)
    .agg(sessions=("user_id", "count"), total_kwh=("energy_kwh", "sum"))
    .sort_values("sessions", ascending=False)
    .head(10)
)
top_by_sessions["user_label"] = top_by_sessions["user_id"].astype(str).apply(lambda x: f"User …{x[-4:]}")

col_u1, col_u2 = st.columns(2)
with col_u1:
    fig_top_energy = px.bar(
        top_by_energy, x="user_label", y="total_kwh",
        color="total_kwh", color_continuous_scale="Purples",
        labels={"user_label": "User", "total_kwh": "Total kWh"},
        title="Top 10 users by energy consumed",
        text=top_by_energy["total_kwh"].apply(lambda x: f"{x:.0f}"),
    )
    fig_top_energy.update_traces(textposition="outside")
    fig_top_energy.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_top_energy, use_container_width=True)
with col_u2:
    fig_top_sessions = px.bar(
        top_by_sessions, x="user_label", y="sessions",
        color="sessions", color_continuous_scale="Blues",
        labels={"user_label": "User", "sessions": "Sessions"},
        title="Top 10 users by session count",
        text=top_by_sessions["sessions"].astype(str),
    )
    fig_top_sessions.update_traces(textposition="outside")
    fig_top_sessions.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_top_sessions, use_container_width=True)

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

# ── AI summary ─────────────────────────────────────────────────────────────────
st.subheader("AI summary")

if len(util_by_office) > 0:
    critical = util_by_office[util_by_office["status"] == "Critical"]["office"].tolist()
    moderate = util_by_office[util_by_office["status"] == "Moderate"]["office"].tolist()
    healthy = util_by_office[util_by_office["status"] == "Healthy"]["office"].tolist()

    top_office = office_sessions.iloc[0]["office"] if len(office_sessions) > 0 else "N/A"
    top_sessions = int(office_sessions.iloc[0]["sessions"]) if len(office_sessions) > 0 else 0

    summary_parts = [
        f"Across the {len(selected_offices)} office(s) selected, **{total_sessions:,} charging sessions** were "
        f"recorded, delivering **{total_energy:,.0f} kWh** of energy to **{unique_users:,} unique employees**."
    ]

    if critical:
        summary_parts.append(
            f"**{', '.join(critical)}** {'is' if len(critical) == 1 else 'are'} running at critical utilization "
            f"(≥70 % during business hours) and should be prioritized for additional charger capacity."
        )
    if moderate:
        summary_parts.append(
            f"**{', '.join(moderate)}** {'shows' if len(moderate) == 1 else 'show'} moderate utilization "
            f"(45–70 %) — monitor these locations as EV adoption continues to grow."
        )
    if healthy:
        summary_parts.append(
            f"**{', '.join(healthy)}** {'is' if len(healthy) == 1 else 'are'} operating at healthy utilization "
            f"levels (<45 %) with sufficient capacity headroom."
        )

    summary_parts.append(
        f"The highest-volume location is **{top_office}** with {top_sessions:,} sessions. "
        "Use the heatmap and peak-hours chart above to identify specific congestion windows and plan "
        "targeted infrastructure investments or demand-smoothing incentives."
    )

    summary_parts.append(
        f"The fleet has avoided an estimated **{co2_saved_kg / 1000:.1f} tonnes of CO\u2082** "
        f"(~{co2_saved_kg / 21:,.0f} tree-years of absorption) and saved an estimated "
        f"**${fleet_cost:,.0f}** in electricity costs at ${cost_per_kwh}/kWh."
    )

    if len(idle_by_office) > 0:
        worst_idle = idle_by_office.iloc[0]
        if worst_idle["avg_idle_hrs"] > 0.5:
            summary_parts.append(
                f"Charging efficiency opportunity: **{worst_idle['office']}** averages "
                f"{worst_idle['avg_idle_hrs']:.1f} hrs of idle time per session and wastes "
                f"{worst_idle['total_idle_hrs']:,.0f} charger-hours in total. "
                "Encouraging employees to unplug promptly after charging would meaningfully improve availability."
            )

    st.info(" ".join(summary_parts))
else:
    st.info("Select at least one office to generate the AI summary.")

with st.expander("Data notes"):
    st.write(
        "Utilization rate is derived from **charger occupancy over time**: each session is spread across "
        "every hour the car stays plugged in (Total Duration), so a car plugged in at 7 AM and unplugged "
        "at 11 AM counts against 7, 8, 9 and 10 AM — not just 7 AM."
    )
    st.write(
        "Charger port count is estimated as the number of distinct Station Name + Port Number combinations "
        "observed in the dataset. This may undercount if some ports never appeared in the export."
    )
    st.write(
        "Note: this export covers only a sample of the site's chargers, so figures reflect **relative demand "
        "patterns**, not absolute charger counts or live telemetry."
    )
    st.write(f"Loaded {len(df):,} charging sessions.")
