import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="PharmaSight Agent", layout="wide", page_icon="💊")

# ── HIDE PLOTLY TOOLBAR & STREAMLIT DEFAULTS ──
st.markdown("""
<style>
    /* Clean metric styling */
    [data-testid="stMetricValue"] { font-size: 2rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
    /* Subtle divider */
    hr { margin: 1.5rem 0; border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──
@st.cache_data
def load_data():
    member = pd.read_csv("dashboard_data.csv")
    pdc = pd.read_csv("member_pdc_scores.csv")
    interventions = pd.read_csv("intervention_recommendations.csv")
    gaps = pd.read_csv("member_gap_analysis.csv")
    
    # Clean member IDs to short readable format
    unique_ids = sorted(member["BENE_ID"].unique())
    id_map = {old: f"MBR-{str(i+1).zfill(4)}" for i, old in enumerate(unique_ids)}
    member["MEMBER_ID"] = member["BENE_ID"].map(id_map)
    interventions["MEMBER_ID"] = interventions["BENE_ID"].map(id_map)
    
    return member, pdc, interventions, gaps

member, pdc, interventions, gaps = load_data()

# ── SIDEBAR ──
st.sidebar.markdown("## PharmaSight Agent")
st.sidebar.caption("Pharmacy claims-driven chronic disease risk intelligence")
st.sidebar.markdown("---")
view = st.sidebar.radio("", [
    "Population Overview",
    "Adherence Deep Dive",
    "Risk & Cost Impact",
    "Intervention Queue"
])
st.sidebar.markdown("---")
st.sidebar.caption("Built on CMS SynPUF 2025 synthetic Medicare data.  \nAdherence metric: PDC (Proportion of Days Covered) — the CMS Star Rating standard.")

# ── ALTAIR THEME ──
RISK_PALETTE = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
THERAPY_PALETTE = {"Cardiovascular": "#3b82f6", "Diabetes": "#ef4444",
                   "Mental_Health": "#a855f7", "Respiratory": "#f59e0b"}

# ══════════════════════════════════════════════════════════════
# VIEW 1: POPULATION OVERVIEW
# ══════════════════════════════════════════════════════════════
if view == "Population Overview":
    st.title("Population Overview")
    st.markdown("A snapshot of the chronic disease cohort — who they are, how many are at risk, and why polypharmacy matters.")
    
    st.markdown("---")

    # KPI row — 4 distinct metrics
    non_adh = member[member["risk_tier"].isin(["High", "Medium"])]
    total_fills = int(pdc["fill_count"].sum())
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Members", f"{len(member):,}")
    col2.metric("Total Rx Fills (2022)", f"{total_fills:,}")
    col3.metric("Non-Adherent", f"{len(non_adh):,}",
                delta=f"{len(non_adh)/len(member)*100:.0f}% of cohort", delta_color="inverse")
    col4.metric("High Risk (PDC < 50%)", f"{len(member[member['risk_tier']=='High']):,}",
                delta=f"{len(member[member['risk_tier']=='High'])/len(member)*100:.0f}% of cohort", delta_color="inverse")

    st.markdown("---")
    st.subheader("Risk Tier Distribution")
    st.markdown("Members are classified into three tiers based on their lowest PDC score across all therapy classes. "
                "**High** = PDC below 50%, **Medium** = 50–79%, **Low** = 80% and above (adherent).")

    tier_data = member["risk_tier"].value_counts().reset_index()
    tier_data.columns = ["Risk Tier", "Members"]
    tier_data["Risk Tier"] = pd.Categorical(tier_data["Risk Tier"], categories=["High", "Medium", "Low"], ordered=True)
    tier_data = tier_data.sort_values("Risk Tier")

    chart = alt.Chart(tier_data).mark_arc(innerRadius=60, outerRadius=130).encode(
        theta=alt.Theta("Members:Q"),
        color=alt.Color("Risk Tier:N", scale=alt.Scale(
            domain=["High", "Medium", "Low"],
            range=["#ef4444", "#f59e0b", "#22c55e"]
        ), legend=alt.Legend(orient="right")),
        tooltip=["Risk Tier", "Members"]
    ).properties(height=300)
    
    text = alt.Chart(tier_data).mark_text(radiusOffset=20, size=14, fontWeight="bold").encode(
        theta=alt.Theta("Members:Q", stack=True),
        text=alt.Text("Members:Q"),
        color=alt.value("#333333")
    )
    
    st.altair_chart(chart + text, use_container_width=True)

    st.markdown("---")
    st.subheader("Polypharmacy Risk")
    st.markdown("Members on **2 or more chronic drug classes** are hospitalized at nearly **double the rate** of single-therapy members. "
                "This is one of the strongest risk signals in the data.")

    multi = member[member["therapy_count"] >= 2]
    single = member[member["therapy_count"] == 1]
    poly_data = pd.DataFrame({
        "Group": ["Single Therapy", "Multi-Therapy (2+)"],
        "Hospitalization Rate": [
            round((single["admission_count"] > 0).mean() * 100, 1),
            round((multi["admission_count"] > 0).mean() * 100, 1)
        ],
        "Members": [len(single), len(multi)]
    })

    chart = alt.Chart(poly_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=80).encode(
        x=alt.X("Group:N", axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("Hospitalization Rate:Q", title="% Hospitalized", scale=alt.Scale(domain=[0, 50])),
        color=alt.Color("Group:N", scale=alt.Scale(
            domain=["Single Therapy", "Multi-Therapy (2+)"],
            range=["#3b82f6", "#ef4444"]
        ), legend=None),
        tooltip=["Group", "Hospitalization Rate", "Members"]
    ).properties(height=300)

    text = chart.mark_text(dy=-15, size=16, fontWeight="bold").encode(
        text=alt.Text("Hospitalization Rate:Q", format=".1f")
    )

    st.altair_chart(chart + text, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VIEW 2: ADHERENCE DEEP DIVE
# ══════════════════════════════════════════════════════════════
elif view == "Adherence Deep Dive":
    st.title("Adherence Deep Dive")
    st.markdown("PDC (Proportion of Days Covered) measures how many days a member had active medication coverage. "
                "CMS considers **80% or above as adherent** — this threshold drives Medicare Star Ratings.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average PDC", f"{pdc['pdc'].mean():.1f}%")
    col2.metric("Adherent (≥ 80%)", f"{(pdc['pdc']>=80).sum():,}",
                delta=f"{(pdc['pdc']>=80).mean()*100:.0f}% of members", delta_color="normal")
    col3.metric("Non-Adherent (< 80%)", f"{(pdc['pdc']<80).sum():,}",
                delta=f"{(pdc['pdc']<80).mean()*100:.0f}% of members", delta_color="inverse")

    st.markdown("---")
    st.subheader("PDC Distribution")
    st.markdown("The bimodal pattern is typical — members are either highly adherent or severely non-adherent. "
                "The red dashed line marks the **80% CMS threshold**.")

    pdc_hist = pdc.copy()
    pdc_hist["pdc_bin"] = pd.cut(pdc_hist["pdc"], bins=range(0, 105, 5), right=False)
    pdc_hist["bin_label"] = pdc_hist["pdc_bin"].apply(lambda x: f"{int(x.left)}–{int(x.right)}%" if pd.notna(x) else "")
    pdc_hist["bin_start"] = pdc_hist["pdc_bin"].apply(lambda x: int(x.left) if pd.notna(x) else 0)
    bin_counts = pdc_hist.groupby(["bin_label", "bin_start"]).size().reset_index(name="count")

    bars = alt.Chart(bin_counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#3b82f6").encode(
        x=alt.X("bin_start:Q", title="PDC (%)", scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=list(range(0, 101, 10)))),
        y=alt.Y("count:Q", title="Members"),
        tooltip=[alt.Tooltip("bin_label:N", title="PDC Range"), alt.Tooltip("count:Q", title="Members")]
    ).properties(height=300)

    rule = alt.Chart(pd.DataFrame({"x": [80]})).mark_rule(
        strokeDash=[6, 4], color="#ef4444", strokeWidth=2
    ).encode(x="x:Q")

    label = alt.Chart(pd.DataFrame({"x": [81], "y": [bin_counts["count"].max() * 0.9], "text": ["80% Threshold"]})).mark_text(
        align="left", color="#ef4444", fontSize=12, fontWeight="bold"
    ).encode(x="x:Q", y="y:Q", text="text:N")

    st.altair_chart(bars + rule + label, use_container_width=True)

    st.markdown("---")
    st.subheader("Adherence Rate by Therapy Class")
    st.markdown("Mental Health and Respiratory have critically low adherence — under 16%. "
                "Diabetes at 41.6% represents the highest-impact intervention opportunity given member volume.")

    adh = pdc.groupby("therapy_class").agg(
        adherent_pct=("pdc", lambda x: round((x >= 80).mean() * 100, 1)),
        members=("BENE_ID", "nunique")
    ).reset_index().sort_values("adherent_pct")

    chart = alt.Chart(adh).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("adherent_pct:Q", title="% Members Adherent", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("therapy_class:N", title=None, sort=alt.EncodingSortField(field="adherent_pct", order="ascending")),
        color=alt.Color("therapy_class:N", scale=alt.Scale(
            domain=list(THERAPY_PALETTE.keys()), range=list(THERAPY_PALETTE.values())
        ), legend=None),
        tooltip=["therapy_class", "adherent_pct", "members"]
    ).properties(height=250)

    text = chart.mark_text(dx=25, size=14, fontWeight="bold").encode(
        text=alt.Text("adherent_pct:Q", format=".1f")
    )

    st.altair_chart(chart + text, use_container_width=True)

    # Gap severity
    if len(gaps) > 0:
        st.markdown("---")
        st.subheader("Medication Gap Severity")
        st.markdown("When a member goes more than **30 days** without filling a chronic medication, it signals potential therapy abandonment. "
                    "Gaps over 90 days mean complete disengagement — these members are the highest priority for outreach.")

        gaps_copy = gaps.copy()
        gaps_copy["Severity"] = pd.cut(gaps_copy["max_gap_days"],
                                        bins=[0, 30, 60, 90, 400],
                                        labels=["< 30 days", "30–60 days", "60–90 days", "> 90 days"])
        sev_counts = gaps_copy["Severity"].value_counts().reindex(
            ["< 30 days", "30–60 days", "60–90 days", "> 90 days"]
        ).fillna(0).reset_index()
        sev_counts.columns = ["Gap Duration", "Members"]

        chart = alt.Chart(sev_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60).encode(
            x=alt.X("Gap Duration:N", sort=["< 30 days", "30–60 days", "60–90 days", "> 90 days"],
                    axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Members:Q", title="Members"),
            color=alt.Color("Gap Duration:N", scale=alt.Scale(
                domain=["< 30 days", "30–60 days", "60–90 days", "> 90 days"],
                range=["#22c55e", "#f59e0b", "#ef4444", "#991b1b"]
            ), legend=None),
            tooltip=["Gap Duration", "Members"]
        ).properties(height=280)

        text = chart.mark_text(dy=-12, size=14, fontWeight="bold").encode(
            text=alt.Text("Members:Q", format=".0f")
        )

        st.altair_chart(chart + text, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# VIEW 3: RISK & COST IMPACT
# ══════════════════════════════════════════════════════════════
elif view == "Risk & Cost Impact":
    st.title("Risk & Cost Impact")
    st.markdown("The financial case for adherence intervention — connecting pharmacy non-adherence to downstream hospitalization spend.")

    st.markdown("---")

    non_adh = member[member["risk_tier"].isin(["High", "Medium"])]
    avoidable = non_adh["total_hosp_cost"].sum() * 0.20

    col1, col2, col3 = st.columns(3)
    col1.metric("Non-Adherent Hosp. Cost", f"${non_adh['total_hosp_cost'].sum():,.0f}")
    col2.metric("Avoidable Cost (20% reduction)", f"${avoidable:,.0f}")
    col3.metric("Savings per Member", f"${avoidable / len(non_adh):,.0f}")

    st.markdown("---")
    st.subheader("Cost per Member by Risk Tier")
    st.markdown("While Low-risk members have higher **total** cost (larger group), the **per-member** cost tells the real story — "
                "High and Medium risk members drive disproportionate spend per person.")

    cost_data = []
    for tier in ["High", "Medium", "Low"]:
        subset = member[member["risk_tier"] == tier]
        hospitalized = subset[subset["admission_count"] > 0]
        cost_data.append({
            "Risk Tier": tier,
            "Cost per Member": round(subset["total_hosp_cost"].sum() / len(subset)),
            "Hospitalization Rate": round((subset["admission_count"] > 0).mean() * 100, 1),
            "Members": len(subset)
        })
    cost_df = pd.DataFrame(cost_data)

    col1, col2 = st.columns(2)

    with col1:
        chart = alt.Chart(cost_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60).encode(
            x=alt.X("Risk Tier:N", sort=["High", "Medium", "Low"], axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Cost per Member:Q", title="Cost per Member ($)"),
            color=alt.Color("Risk Tier:N", scale=alt.Scale(
                domain=["High", "Medium", "Low"], range=["#ef4444", "#f59e0b", "#22c55e"]
            ), legend=None),
            tooltip=["Risk Tier", "Cost per Member", "Members"]
        ).properties(height=320, title="Per-Member Hospitalization Cost")

        text = chart.mark_text(dy=-12, size=14, fontWeight="bold").encode(
            text=alt.Text("Cost per Member:Q", format="$,.0f")
        )
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        chart = alt.Chart(cost_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=60).encode(
            x=alt.X("Risk Tier:N", sort=["High", "Medium", "Low"], axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Hospitalization Rate:Q", title="% Hospitalized"),
            color=alt.Color("Risk Tier:N", scale=alt.Scale(
                domain=["High", "Medium", "Low"], range=["#ef4444", "#f59e0b", "#22c55e"]
            ), legend=None),
            tooltip=["Risk Tier", "Hospitalization Rate", "Members"]
        ).properties(height=320, title="Hospitalization Rate")

        text = chart.mark_text(dy=-12, size=14, fontWeight="bold").encode(
            text=alt.Text("Hospitalization Rate:Q", format=".1f")
        )
        st.altair_chart(chart + text, use_container_width=True)

    st.markdown("---")
    st.markdown("**Methodology:** Avoidable cost is estimated at a conservative 20% reduction in hospitalization spend "
                "for non-adherent members, based on published research linking PDC improvement to reduced inpatient utilization. "
                "Real-world savings vary by therapy class and intervention type.")

# ══════════════════════════════════════════════════════════════
# VIEW 4: INTERVENTION QUEUE
# ══════════════════════════════════════════════════════════════
elif view == "Intervention Queue":
    st.title("Intervention Queue")
    st.markdown("A prioritized action list for care managers — every member here has a PDC below 80% and a specific set of recommended interventions "
                "based on their therapy class, gap pattern, and hospitalization history.")

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        urgency_opts = sorted(interventions["urgency"].dropna().unique().tolist())
        urgency_filter = st.multiselect("Urgency", urgency_opts, default=urgency_opts)
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
    col1.metric("Members in Queue", f"{len(filtered):,}")
    critical_n = len(filtered[filtered["urgency"] == "Critical"])
    col2.metric("Critical", f"{critical_n:,}",
                delta="Immediate outreach needed" if critical_n > 0 else None, delta_color="inverse")
    col3.metric("Cost at Risk", f"${filtered['total_hosp_cost'].fillna(0).sum():,.0f}")

    st.markdown("---")

    if len(filtered) == 0:
        st.info("No members match the selected filters. Try expanding the filters above.")
    else:
        # Urgency breakdown
        st.subheader("Urgency Breakdown")
        st.markdown("**Critical** = hospitalized + non-adherent (needs immediate pharmacist outreach). "
                    "**Urgent** = PDC below 50% (medication therapy management). "
                    "**Elevated** = PDC 50–79% (automated reminders + 90-day supply conversion).")

        urg_data = filtered["urgency"].value_counts().reset_index()
        urg_data.columns = ["Urgency", "Members"]

        chart = alt.Chart(urg_data).mark_arc(innerRadius=60, outerRadius=120).encode(
            theta=alt.Theta("Members:Q"),
            color=alt.Color("Urgency:N", scale=alt.Scale(
                domain=["Critical", "Urgent", "Elevated"],
                range=["#ef4444", "#f59e0b", "#3b82f6"]
            ), legend=alt.Legend(orient="right")),
            tooltip=["Urgency", "Members"]
        ).properties(height=280)

        text = alt.Chart(urg_data).mark_text(radiusOffset=20, size=14, fontWeight="bold").encode(
            theta=alt.Theta("Members:Q", stack=True),
            text=alt.Text("Members:Q"),
            color=alt.value("#333333")
        )

        st.altair_chart(chart + text, use_container_width=True)

        st.markdown("---")
        st.subheader("Priority Member List")
        st.markdown("Sorted by urgency — Critical first. Click any row's expander below for full intervention details.")

        # Clean table
        show = filtered[["MEMBER_ID", "worst_pdc", "therapies", "admission_count",
                         "total_hosp_cost", "urgency"]].copy()
        show.columns = ["Member", "PDC %", "Therapies", "Admissions", "Hosp. Cost ($)", "Urgency"]
        show["Hosp. Cost ($)"] = show["Hosp. Cost ($)"].fillna(0).apply(lambda x: f"${x:,.0f}")
        show["Admissions"] = show["Admissions"].fillna(0).astype(int)
        priority = {"Critical": 0, "Urgent": 1, "Elevated": 2}
        show["_sort"] = show["Urgency"].map(priority)
        show = show.sort_values("_sort").drop(columns="_sort").head(20)

        st.dataframe(show, use_container_width=True, height=400, hide_index=True)

        # Top member cards
        st.markdown("---")
        st.subheader("Top Priority Members")

        cards = filtered.copy()
        cards["_sort"] = cards["urgency"].map(priority)
        cards = cards.sort_values("_sort").head(5)

        for _, row in cards.iterrows():
            icon = "🔴" if row["urgency"] == "Critical" else "🟡" if row["urgency"] == "Urgent" else "🔵"
            with st.expander(f'{icon} {row["MEMBER_ID"]} — {row["urgency"]} — PDC: {row["worst_pdc"]}%'):
                c1, c2, c3 = st.columns(3)
                c1.metric("PDC", f'{row["worst_pdc"]}%')
                c2.metric("Admissions", int(row["admission_count"]) if pd.notna(row["admission_count"]) else 0)
                cost = row["total_hosp_cost"] if pd.notna(row["total_hosp_cost"]) else 0
                c3.metric("Hosp. Cost", f"${cost:,.0f}")
                st.markdown(f'**Therapies:** {row["therapies"]}')
                intv = str(row["interventions"])
                if intv and intv != "nan":
                    st.markdown("**Recommended Actions:**")
                    for action in intv.split(" | "):
                        if action.strip():
                            st.markdown(f"- {action.strip()}")
