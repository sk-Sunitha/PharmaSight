import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="PharmaSight Agent", layout="wide", page_icon="💊")

# ══════════════════════════════════════════════════════════════
# STYLING — compact layout, proper hierarchy, bordered KPI cards
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Tighten top padding so content is visible at a glance */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

    /* App title hierarchy */
    h1 { font-size: 1.75rem !important; margin: 0 0 0.25rem 0 !important; font-weight: 700; }
    h2 { font-size: 1.35rem !important; margin: 0.5rem 0 0.4rem 0 !important; font-weight: 600; }
    h3 { font-size: 1.1rem !important; margin: 0 0 0.3rem 0 !important; font-weight: 600; }
    h4 { font-size: 1rem !important; margin: 0 0 0.4rem 0 !important; font-weight: 600; color: #111827; }

    hr { margin: 0.8rem 0 !important; border-color: #e5e7eb; }

    /* KPI cards — wrap the default Streamlit metric in a bordered pill */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #6b7280; font-weight: 500; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* Chart container cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
    }

    /* Native tab styling — pill/segmented look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: transparent;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 6px;
        padding: 0.45rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #e5e7eb; }
    .stTabs [aria-selected="true"] {
        background: #2c5282 !important;
        color: white !important;
        border-color: #2c5282 !important;
    }

    /* Subtitle */
    .app-subtitle { color: #6b7280; font-size: 0.82rem; margin: 0 0 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
def load_data():
    member = pd.read_csv("dashboard_data.csv")
    pdc = pd.read_csv("member_pdc_scores.csv")
    interventions = pd.read_csv("intervention_recommendations.csv")
    gaps = pd.read_csv("member_gap_analysis.csv")

    unique_ids = sorted(member["BENE_ID"].unique())
    id_map = {old: f"MBR-{str(i+1).zfill(4)}" for i, old in enumerate(unique_ids)}
    member["MEMBER_ID"] = member["BENE_ID"].map(id_map)
    interventions["MEMBER_ID"] = interventions["BENE_ID"].map(id_map)

    return member, pdc, interventions, gaps

member, pdc, interventions, gaps = load_data()

# ══════════════════════════════════════════════════════════════
# COLOR PALETTE
# ══════════════════════════════════════════════════════════════
RISK_DOMAIN = ["High", "Medium", "Low"]
RISK_RANGE = ["#e07a5f", "#f2b56b", "#6fa8c7"]
THERAPY_PALETTE = {
    "Cardiovascular": "#5b8fb9", "Diabetes": "#e07a5f",
    "Mental_Health": "#9b8ec4", "Respiratory": "#e8a838"
}
URGENCY_DOMAIN = ["Critical", "Urgent", "Elevated"]
URGENCY_RANGE = ["#d45d5d", "#e8a838", "#5b8fb9"]

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# PharmaSight Agent")
st.markdown('<p class="app-subtitle">Pharmacy claims-driven chronic disease risk intelligence · '
            'CMS SynPUF synthetic Medicare data · Adherence metric: PDC (CMS Star Rating standard)</p>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TOP NAVIGATION — native tabs (pill/segmented style via CSS)
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Population Overview",
    "Adherence Deep Dive",
    "Risk & Cost Impact",
    "Intervention Queue",
])

# ══════════════════════════════════════════════════════════════
# VIEW 1: POPULATION OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    non_adh = member[member["risk_tier"].isin(["High", "Medium"])]
    total_fills = int(pdc["fill_count"].sum())
    non_adh_pct = round(len(non_adh) / len(member) * 100)
    high_n = len(member[member["risk_tier"] == "High"])
    high_pct = round(high_n / len(member) * 100)

    # KPI row — plain captions, no misleading arrows
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Members", f"{len(member):,}")
    c2.metric("Total Rx Fills", f"{total_fills:,}")
    c3.metric("Non-Adherent", f"{len(non_adh):,}", f"{non_adh_pct}% of cohort", delta_color="off")
    c4.metric("High Risk (PDC < 50%)", f"{high_n:,}", f"{high_pct}% of cohort", delta_color="off")

    st.markdown("")  # small spacer

    col1, col2 = st.columns(2)

    # ── Risk Tier Distribution (donut with % labels) ──
    with col1:
        with st.container(border=True):
            st.markdown("#### Risk Tier Distribution")

            tier_data = member["risk_tier"].value_counts().reset_index()
            tier_data.columns = ["Risk Tier", "Members"]
            tier_data["Risk Tier"] = pd.Categorical(tier_data["Risk Tier"], categories=RISK_DOMAIN, ordered=True)
            tier_data = tier_data.sort_values("Risk Tier")
            tier_data["Pct"] = (tier_data["Members"] / tier_data["Members"].sum() * 100).round(1)
            tier_data["PctLabel"] = tier_data["Pct"].apply(lambda p: f"{p}%")

            # Donut
            pie = alt.Chart(tier_data).mark_arc(innerRadius=55, outerRadius=105).encode(
                theta=alt.Theta("Members:Q", stack=True),
                color=alt.Color("Risk Tier:N",
                                scale=alt.Scale(domain=RISK_DOMAIN, range=RISK_RANGE),
                                legend=alt.Legend(title=None, orient="bottom", direction="horizontal")),
                tooltip=["Risk Tier", "Members",
                         alt.Tooltip("Pct:Q", title="% of Total", format=".1f")]
            )

            # % labels on each arc
            labels = alt.Chart(tier_data).mark_text(
                radius=130, size=12, fontWeight="bold", color="#111827"
            ).encode(
                theta=alt.Theta("Members:Q", stack=True),
                text="PctLabel:N"
            )

            chart = (pie + labels).properties(height=230).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

            st.caption(f"**{non_adh_pct}% of the cohort is non-adherent** "
                       f"(High + Medium risk) — this is where intervention spend should concentrate.")

    # ── Hospitalization Rate by Risk Tier ──
    with col2:
        with st.container(border=True):
            st.markdown("#### Hospitalization Rate by Risk Tier")

            rate_data = []
            for tier in RISK_DOMAIN:
                subset = member[member["risk_tier"] == tier]
                rate_data.append({
                    "Risk Tier": tier,
                    "Hospitalization Rate": round((subset["admission_count"] > 0).mean() * 100, 1),
                    "Members": len(subset),
                })
            rate_df = pd.DataFrame(rate_data)
            ymax_rate = min(100, int(rate_df["Hospitalization Rate"].max() * 1.15))

            chart = alt.Chart(rate_df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=45
            ).encode(
                x=alt.X("Risk Tier:N", sort=RISK_DOMAIN, axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y("Hospitalization Rate:Q", title="% Hospitalized",
                        scale=alt.Scale(domain=[0, ymax_rate])),
                color=alt.Color("Risk Tier:N",
                                scale=alt.Scale(domain=RISK_DOMAIN, range=RISK_RANGE),
                                legend=None),
                tooltip=["Risk Tier", "Hospitalization Rate", "Members"],
            ).properties(height=230)

            text = chart.mark_text(dy=-10, size=12, fontWeight="bold").encode(
                text=alt.Text("Hospitalization Rate:Q", format=".1f")
            )
            st.altair_chart(chart + text, use_container_width=True)

            st.caption("High-risk members are hospitalized at the highest rate — confirming the link "
                       "between pharmacy non-adherence and inpatient utilization.")

# ══════════════════════════════════════════════════════════════
# VIEW 2: ADHERENCE DEEP DIVE
# ══════════════════════════════════════════════════════════════
with tab2:
    avg_pdc = pdc["pdc"].mean()
    adh_n = (pdc["pdc"] >= 80).sum()
    non_n = (pdc["pdc"] < 80).sum()
    adh_pct = round((pdc["pdc"] >= 80).mean() * 100)
    non_pct = round((pdc["pdc"] < 80).mean() * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average PDC", f"{avg_pdc:.1f}%")
    c2.metric("Adherent (≥ 80%)", f"{adh_n:,}", f"{adh_pct}% of PDC scores", delta_color="off")
    c3.metric("Non-Adherent (< 80%)", f"{non_n:,}", f"{non_pct}% of PDC scores", delta_color="off")

    st.markdown("")

    col1, col2 = st.columns(2)

    # ── PDC Distribution ──
    with col1:
        with st.container(border=True):
            st.markdown("#### PDC Distribution")

            pdc_hist = pdc.copy()
            if pdc_hist["pdc"].max() <= 1.5:
                pdc_hist["pdc"] = pdc_hist["pdc"] * 100
            pdc_hist["pdc_bin"] = pd.cut(pdc_hist["pdc"],
                                         bins=list(range(0, 105, 5)),
                                         right=True, include_lowest=True)
            pdc_hist["bin_label"] = pdc_hist["pdc_bin"].apply(
                lambda x: f"{int(x.left)}–{int(x.right)}%" if pd.notna(x) else ""
            )
            pdc_hist["bin_start"] = pdc_hist["pdc_bin"].apply(
                lambda x: int(x.left) if pd.notna(x) else 0
            )
            bin_counts = pdc_hist.groupby(["bin_label", "bin_start"], observed=True).size().reset_index(name="count")
            ymax = int(bin_counts["count"].max() * 1.15)

            bars = alt.Chart(bin_counts).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#5b8fb9"
            ).encode(
                x=alt.X("bin_start:Q", title="PDC (%)",
                        scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=list(range(0, 101, 20)))),
                y=alt.Y("count:Q", title="Members", scale=alt.Scale(domain=[0, ymax])),
                tooltip=[alt.Tooltip("bin_label:N", title="PDC Range"),
                         alt.Tooltip("count:Q", title="Members")],
            ).properties(height=230)

            rule = alt.Chart(pd.DataFrame({"x": [80]})).mark_rule(
                strokeDash=[6, 4], color="#d45d5d", strokeWidth=2
            ).encode(x="x:Q")

            label = alt.Chart(pd.DataFrame({"x": [81], "y": [ymax * 0.9], "text": ["80%"]})).mark_text(
                align="left", color="#d45d5d", fontSize=11, fontWeight="bold"
            ).encode(x="x:Q", y="y:Q", text="text:N")

            st.altair_chart(bars + rule + label, use_container_width=True)
            st.caption("Bimodal — a cluster of non-adherent members between 30–80% PDC, and ~890 fully "
                       "adherent at 95–100%. Dashed line = 80% CMS threshold.")

    # ── Adherence by Therapy Class ──
    with col2:
        with st.container(border=True):
            st.markdown("#### Adherence Rate by Therapy Class")

            adh = pdc.groupby("therapy_class").agg(
                adherent_pct=("pdc", lambda x: round((x >= 80).mean() * 100, 1)),
                members=("BENE_ID", "nunique")
            ).reset_index().sort_values("adherent_pct")

            chart = alt.Chart(adh).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4
            ).encode(
                x=alt.X("adherent_pct:Q", title="% Members Adherent",
                        scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("therapy_class:N", title=None,
                        sort=alt.EncodingSortField(field="adherent_pct", order="ascending")),
                color=alt.Color("therapy_class:N",
                                scale=alt.Scale(domain=list(THERAPY_PALETTE.keys()),
                                                range=list(THERAPY_PALETTE.values())),
                                legend=None),
                tooltip=["therapy_class", "adherent_pct", "members"]
            ).properties(height=230)

            text = chart.mark_text(dx=6, size=12, fontWeight="bold", align="left").encode(
                text=alt.Text("adherent_pct:Q", format=".1f")
            )
            st.altair_chart(chart + text, use_container_width=True)

            st.caption("Respiratory lowest at 31.5%, Diabetes 38.5%, Cardiovascular leads at 49.1% — "
                       "all short of the 80% CMS threshold.")

# ══════════════════════════════════════════════════════════════
# VIEW 3: RISK & COST IMPACT
# ══════════════════════════════════════════════════════════════
with tab3:
    non_adh = member[member["risk_tier"].isin(["High", "Medium"])]
    non_adh_cost = non_adh["total_hosp_cost"].sum()
    avoidable = non_adh_cost * 0.20

    c1, c2, c3 = st.columns(3)
    c1.metric("Non-Adherent Hosp. Cost", f"${non_adh_cost/1e6:.1f}M")
    c2.metric("Avoidable (20% reduction)", f"${avoidable/1e6:.1f}M")
    c3.metric("Savings per Member", f"${avoidable / len(non_adh):,.0f}")

    st.markdown("")

    with st.container(border=True):
        st.markdown("#### Per-Member Hospitalization Cost by Risk Tier")

        cost_data = []
        for tier in RISK_DOMAIN:
            subset = member[member["risk_tier"] == tier]
            cost_data.append({
                "Risk Tier": tier,
                "Cost per Member": round(subset["total_hosp_cost"].sum() / len(subset)),
                "Members": len(subset),
            })
        cost_df = pd.DataFrame(cost_data)

        chart = alt.Chart(cost_df).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=55
        ).encode(
            x=alt.X("Risk Tier:N", sort=RISK_DOMAIN, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Cost per Member:Q", title="Cost per Member ($)"),
            color=alt.Color("Risk Tier:N",
                            scale=alt.Scale(domain=RISK_DOMAIN, range=RISK_RANGE),
                            legend=None),
            tooltip=["Risk Tier", "Cost per Member", "Members"],
        ).properties(height=260)

        text = chart.mark_text(dy=-10, size=12, fontWeight="bold").encode(
            text=alt.Text("Cost per Member:Q", format="$,.0f")
        )
        st.altair_chart(chart + text, use_container_width=True)

        st.caption("High-risk members have the highest per-member cost. Low > Medium is a "
                   "known Medicare survival artifact — adherent members live longer and accumulate "
                   "more inpatient utilization over time.")

    st.caption("Methodology: Avoidable cost estimated at a conservative 20% hospitalization-spend "
               "reduction for non-adherent members (published PDC-improvement → inpatient-reduction evidence).")

# ══════════════════════════════════════════════════════════════
# VIEW 4: INTERVENTION QUEUE
# ══════════════════════════════════════════════════════════════
with tab4:
    # Filters
    f1, f2 = st.columns(2)
    with f1:
        urgency_opts = sorted(interventions["urgency"].dropna().unique().tolist())
        urgency_filter = st.multiselect("Urgency", urgency_opts, default=urgency_opts)
    with f2:
        all_therapies = set()
        for val in interventions["therapies"].dropna():
            for t in str(val).split(", "):
                all_therapies.add(t.strip())
        therapy_filter = st.multiselect("Therapy Class", sorted(all_therapies), default=sorted(all_therapies))

    filtered = interventions[interventions["urgency"].isin(urgency_filter)].copy()
    if therapy_filter:
        filtered = filtered[filtered["therapies"].apply(
            lambda x: any(t in str(x) for t in therapy_filter)
        )]

    c1, c2, c3 = st.columns(3)
    c1.metric("Members in Queue", f"{len(filtered):,}")
    critical_n = len(filtered[filtered["urgency"] == "Critical"])
    c2.metric("Critical", f"{critical_n:,}",
              "Immediate outreach needed" if critical_n > 0 else None, delta_color="off")
    c3.metric("Cost at Risk", f"${filtered['total_hosp_cost'].fillna(0).sum()/1e6:.1f}M")

    st.markdown("")

    if len(filtered) == 0:
        st.info("No members match the selected filters. Try expanding the filters above.")
    else:
        # Urgency breakdown
        with st.container(border=True):
            st.markdown("#### Urgency Breakdown")
            st.caption("Critical = hospitalized and non-adherent (PDC < 80%) · "
                       "Urgent = PDC < 50% (not hospitalized) · "
                       "Elevated = PDC 50–79% (not hospitalized)")

            urg_data = filtered["urgency"].value_counts().reset_index()
            urg_data.columns = ["Urgency", "Members"]
            urg_data["Pct"] = (urg_data["Members"] / urg_data["Members"].sum() * 100).round(1)
            urg_data["Urgency"] = pd.Categorical(urg_data["Urgency"], categories=URGENCY_DOMAIN, ordered=True)
            urg_data = urg_data.sort_values("Urgency")
            ymax_urg = int(urg_data["Members"].max() * 1.15)

            chart = alt.Chart(urg_data).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=45
            ).encode(
                x=alt.X("Urgency:N", sort=URGENCY_DOMAIN,
                        axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y("Members:Q", title="Members",
                        scale=alt.Scale(domain=[0, ymax_urg])),
                color=alt.Color("Urgency:N",
                                scale=alt.Scale(domain=URGENCY_DOMAIN, range=URGENCY_RANGE),
                                legend=None),
                tooltip=["Urgency", "Members",
                         alt.Tooltip("Pct:Q", title="% of Total", format=".1f")],
            ).properties(height=220)

            text = chart.mark_text(dy=-10, size=12, fontWeight="bold").encode(
                text=alt.Text("Members:Q", format=",")
            )
            st.altair_chart(chart + text, use_container_width=True)

        # Priority list
        with st.container(border=True):
            st.markdown("#### Priority Member List")
            st.caption("Top 20 members, sorted by urgency — Critical first")

            show = filtered[["MEMBER_ID", "worst_pdc", "therapies", "admission_count",
                             "total_hosp_cost", "urgency"]].copy()
            show.columns = ["Member", "PDC %", "Therapies", "Admissions", "Hosp. Cost ($)", "Urgency"]
            show["Hosp. Cost ($)"] = show["Hosp. Cost ($)"].fillna(0).apply(lambda x: f"${x:,.0f}")
            show["Admissions"] = show["Admissions"].fillna(0).astype(int)
            priority = {"Critical": 0, "Urgent": 1, "Elevated": 2}
            show["_sort"] = show["Urgency"].map(priority)
            show = show.sort_values("_sort").drop(columns="_sort").head(20)

            st.dataframe(show, use_container_width=True, height=320, hide_index=True)

        # Top priority member cards
        st.markdown("##### Top Priority Members")

        cards = filtered.copy()
        cards["_sort"] = cards["urgency"].map(priority)
        cards = cards.sort_values("_sort").head(5)

        for _, row in cards.iterrows():
            icon = "🔴" if row["urgency"] == "Critical" else "🟡" if row["urgency"] == "Urgent" else "🔵"
            with st.expander(f'{icon} {row["MEMBER_ID"]} — {row["urgency"]} — PDC: {row["worst_pdc"]}%'):
                c1, c2, c3 = st.columns(3)
                c1.metric("PDC", f'{row["worst_pdc"]}%')
                c2.metric("Admissions",
                          int(row["admission_count"]) if pd.notna(row["admission_count"]) else 0)
                cost = row["total_hosp_cost"] if pd.notna(row["total_hosp_cost"]) else 0
                c3.metric("Hosp. Cost", f"${cost:,.0f}")
                st.markdown(f'**Therapies:** {row["therapies"]}')
                intv = str(row["interventions"])
                if intv and intv != "nan":
                    st.markdown("**Recommended Actions:**")
                    for action in intv.split(" | "):
                        if action.strip():
                            st.markdown(f"- {action.strip()}")
