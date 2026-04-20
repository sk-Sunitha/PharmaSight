import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PharmaSight", layout="wide", page_icon="💊")

# ── LOAD DATA ──
@st.cache_data
def load_data():
    member = pd.read_csv("dashboard_data.csv")
    pdc = pd.read_csv("member_pdc_scores.csv")
    interventions = pd.read_csv("intervention_recommendations.csv")
    gaps = pd.read_csv("member_gap_analysis.csv")
    return member, pdc, interventions, gaps

member, pdc, interventions, gaps = load_data()

# ── SIDEBAR ──
st.sidebar.title("PharmaSight")
st.sidebar.markdown("Chronic Disease Risk Agent")
st.sidebar.markdown("---")
view = st.sidebar.radio("View", [
    "Population Overview",
    "Adherence Deep Dive",
    "Risk & Cost Impact",
    "Intervention Queue"
])

# ── COLORS ──
RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
THERAPY_COLORS = {
    "Cardiovascular": "#3b82f6", "Diabetes": "#ef4444",
    "Mental_Health": "#a855f7", "Respiratory": "#f59e0b"
}

# ══════════════════════════════════════════════════════════════
# VIEW 1: POPULATION OVERVIEW
# ══════════════════════════════════════════════════════════════
if view == "Population Overview":
    st.title("Population Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Members", f"{len(member):,}")
    col2.metric("On Chronic Meds", f"{len(member):,}")
    non_adh_count = len(member[member["risk_tier"].isin(["High", "Medium"])])
    col3.metric("Non-Adherent", f"{non_adh_count:,}")
    col4.metric("High Risk", f"{len(member[member['risk_tier']=='High']):,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = member["risk_tier"].value_counts().reindex(["High", "Medium", "Low"])
        fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                     color=tier_counts.index, color_discrete_map=RISK_COLORS, hole=0.4)
        fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Polypharmacy Risk")
        multi = member[member["therapy_count"] >= 2]
        single = member[member["therapy_count"] == 1]
        multi_hosp = (multi["admission_count"] > 0).mean() * 100
        single_hosp = (single["admission_count"] > 0).mean() * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Single Therapy", x=["Hospitalization Rate %"], y=[round(single_hosp, 1)],
                            marker_color="#3b82f6", text=[f"{single_hosp:.1f}%"], textposition="outside"))
        fig.add_trace(go.Bar(name="Multi-Therapy (2+)", x=["Hospitalization Rate %"], y=[round(multi_hosp, 1)],
                            marker_color="#ef4444", text=[f"{multi_hosp:.1f}%"], textposition="outside"))
        fig.update_layout(height=350, yaxis_title="", barmode="group",
                         margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VIEW 2: ADHERENCE DEEP DIVE
# ══════════════════════════════════════════════════════════════
elif view == "Adherence Deep Dive":
    st.title("Adherence Deep Dive")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average PDC", f"{pdc['pdc'].mean():.1f}%")
    col2.metric("Adherent (≥80%)", f"{(pdc['pdc']>=80).sum():,}")
    col3.metric("Non-Adherent (<80%)", f"{(pdc['pdc']<80).sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PDC Distribution")
        fig = px.histogram(pdc, x="pdc", nbins=20, color_discrete_sequence=["#3b82f6"])
        fig.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="80% Threshold")
        fig.update_layout(height=350, xaxis_title="PDC (%)", yaxis_title="Members",
                         margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Adherence Rate by Therapy Class")
        adh = pdc.groupby("therapy_class").agg(
            adherent_pct=("pdc", lambda x: round((x >= 80).mean() * 100, 1))
        ).reset_index().sort_values("adherent_pct")

        fig = px.bar(adh, x="adherent_pct", y="therapy_class", orientation="h",
                     color="therapy_class", color_discrete_map=THERAPY_COLORS,
                     text="adherent_pct")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=350, xaxis_title="% Adherent", yaxis_title="",
                         showlegend=False, margin=dict(t=20, b=20, l=20, r=100))
        st.plotly_chart(fig, use_container_width=True)

    # Gap analysis — single clean chart
    if len(gaps) > 0:
        st.subheader("Medication Gap Severity")
        gaps_copy = gaps.copy()
        gaps_copy["gap_bucket"] = pd.cut(gaps_copy["max_gap_days"],
                                          bins=[0, 30, 60, 90, 400],
                                          labels=["< 30 days", "30–60 days", "60–90 days", "> 90 days"])
        bucket_counts = gaps_copy["gap_bucket"].value_counts().reindex(
            ["< 30 days", "30–60 days", "60–90 days", "> 90 days"]
        ).fillna(0)
        bucket_colors = ["#22c55e", "#f59e0b", "#ef4444", "#991b1b"]

        fig = go.Figure(go.Bar(
            x=bucket_counts.index, y=bucket_counts.values,
            marker_color=bucket_colors,
            text=bucket_counts.values.astype(int), textposition="outside"
        ))
        fig.update_layout(height=300, xaxis_title="Longest Gap Between Fills",
                         yaxis_title="Members", margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VIEW 3: RISK & COST IMPACT
# ══════════════════════════════════════════════════════════════
elif view == "Risk & Cost Impact":
    st.title("Risk & Cost Impact")

    non_adh = member[member["risk_tier"].isin(["High", "Medium"])]
    avoidable = non_adh["total_hosp_cost"].sum() * 0.20

    col1, col2, col3 = st.columns(3)
    col1.metric("Non-Adherent Hosp. Cost", f"${non_adh['total_hosp_cost'].sum():,.0f}")
    col2.metric("Avoidable Cost (20%)", f"${avoidable:,.0f}")
    col3.metric("Per Member Savings", f"${avoidable / len(non_adh):,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hospitalization Rate by Risk Tier")
        hosp_data = []
        for tier in ["High", "Medium", "Low"]:
            subset = member[member["risk_tier"] == tier]
            rate = (subset["admission_count"] > 0).mean() * 100
            hosp_data.append({"tier": tier, "rate": round(rate, 1)})
        hosp_df = pd.DataFrame(hosp_data)

        fig = px.bar(hosp_df, x="tier", y="rate", color="tier",
                     color_discrete_map=RISK_COLORS, text="rate")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=350, xaxis_title="Risk Tier", yaxis_title="% Hospitalized",
                         showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total Cost by Risk Tier")
        cost_data = []
        for tier in ["High", "Medium", "Low"]:
            subset = member[member["risk_tier"] == tier]
            cost_data.append({"tier": tier, "cost": round(subset["total_hosp_cost"].sum())})
        cost_df = pd.DataFrame(cost_data)

        fig = px.bar(cost_df, x="tier", y="cost", color="tier",
                     color_discrete_map=RISK_COLORS, text="cost")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=350, xaxis_title="Risk Tier", yaxis_title="Total Cost ($)",
                         showlegend=False, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VIEW 4: INTERVENTION QUEUE
# ══════════════════════════════════════════════════════════════
elif view == "Intervention Queue":
    st.title("Intervention Queue")

    # Filters in columns
    col1, col2 = st.columns(2)
    with col1:
        urgency_filter = st.multiselect("Urgency", ["Critical", "Urgent", "Elevated"],
                                         default=["Critical", "Urgent", "Elevated"])
    with col2:
        all_therapies = set()
        for val in interventions["therapies"].dropna():
            for t in str(val).split(", "):
                all_therapies.add(t.strip())
        therapy_filter = st.multiselect("Therapy Class", sorted(all_therapies), default=sorted(all_therapies))

    # Apply filters
    filtered = interventions[interventions["urgency"].isin(urgency_filter)].copy()
    if therapy_filter:
        filtered = filtered[filtered["therapies"].apply(
            lambda x: any(t in str(x) for t in therapy_filter)
        )]

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Members in Queue", len(filtered))
    col2.metric("Critical", len(filtered[filtered["urgency"] == "Critical"]))
    col3.metric("Cost at Risk", f"${filtered['total_hosp_cost'].fillna(0).sum():,.0f}")

    st.markdown("---")

    if len(filtered) == 0:
        st.warning("No members match the selected filters.")
    else:
        # Urgency pie + table side by side
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("By Urgency")
            urg_counts = filtered["urgency"].value_counts()
            urg_colors = {"Critical": "#ef4444", "Urgent": "#f59e0b", "Elevated": "#3b82f6"}
            fig = px.pie(values=urg_counts.values, names=urg_counts.index,
                         color=urg_counts.index, color_discrete_map=urg_colors, hole=0.4)
            fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Priority List")
            show_cols = ["BENE_ID", "worst_pdc", "therapies", "admission_count",
                        "total_hosp_cost", "urgency"]
            priority = {"Critical": 0, "Urgent": 1, "Elevated": 2}
            sorted_df = filtered[show_cols].copy()
            sorted_df = sorted_df.sort_values("urgency", key=lambda x: x.map(priority))
            sorted_df.columns = ["Member ID", "PDC %", "Therapies", "Admissions", "Hosp Cost", "Urgency"]
            st.dataframe(sorted_df.head(15), use_container_width=True, height=320)

        # Top 5 member cards
        st.subheader("Top Priority Members")
        priority_map = {"Critical": 0, "Urgent": 1, "Elevated": 2}
        cards = filtered.copy()
        cards["_sort"] = cards["urgency"].map(priority_map)
        cards = cards.sort_values("_sort").head(5)

        for _, row in cards.iterrows():
            icon = "🔴" if row["urgency"] == "Critical" else "🟡" if row["urgency"] == "Urgent" else "🔵"
            label = f"{icon} Member {row['BENE_ID']} — {row['urgency']} — PDC: {row['worst_pdc']}%"
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                c1.metric("PDC", f"{row['worst_pdc']}%")
                c2.metric("Admissions", int(row["admission_count"]) if pd.notna(row["admission_count"]) else 0)
                cost = row["total_hosp_cost"] if pd.notna(row["total_hosp_cost"]) else 0
                c3.metric("Hosp. Cost", f"${cost:,.0f}")
                st.markdown(f"**Therapies:** {row['therapies']}")
                interventions_text = str(row["interventions"])
                if interventions_text and interventions_text != "nan":
                    st.markdown("**Recommended Actions:**")
                    for action in interventions_text.split(" | "):
                        if action.strip():
                            st.markdown(f"- {action.strip()}")

# ── FOOTER ──
st.sidebar.markdown("---")
st.sidebar.markdown("**PharmaSight v1.0**")
st.sidebar.markdown("CMS SynPUF 2025 data")