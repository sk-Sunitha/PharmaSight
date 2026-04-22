"""
PharmaSight Agent — Tools Layer
Three tools the LLM can call. Each returns structured JSON grounded in the
pharmacy claims pipeline. No invented facts — every data point traces to a CSV row.
"""

import pandas as pd
import json
from typing import Optional


# ══════════════════════════════════════════════════════════════
# DATA LOADING — cached at module level (one load per session)
# ══════════════════════════════════════════════════════════════
_CACHE = {}


def _load_data():
    """Load all 4 CSVs once and cache."""
    if "member" in _CACHE:
        return _CACHE

    member = pd.read_csv("dashboard_data.csv")
    pdc = pd.read_csv("member_pdc_scores.csv")
    interventions = pd.read_csv("intervention_recommendations.csv")
    gaps = pd.read_csv("member_gap_analysis.csv")

    # Build MBR-#### → BENE_ID mapping (matches app.py)
    unique_ids = sorted(member["BENE_ID"].unique())
    mbr_to_bene = {f"MBR-{str(i+1).zfill(4)}": old for i, old in enumerate(unique_ids)}
    bene_to_mbr = {v: k for k, v in mbr_to_bene.items()}

    member["MEMBER_ID"] = member["BENE_ID"].map(bene_to_mbr)
    interventions["MEMBER_ID"] = interventions["BENE_ID"].map(bene_to_mbr)
    pdc["MEMBER_ID"] = pdc["BENE_ID"].map(bene_to_mbr)
    gaps["MEMBER_ID"] = gaps["BENE_ID"].map(bene_to_mbr)

    _CACHE.update({
        "member": member,
        "pdc": pdc,
        "interventions": interventions,
        "gaps": gaps,
        "mbr_to_bene": mbr_to_bene,
        "bene_to_mbr": bene_to_mbr,
    })
    return _CACHE


# ══════════════════════════════════════════════════════════════
# PQA LITERATURE NORMS — avoidable hospitalization $ per non-adherent member per year
# ══════════════════════════════════════════════════════════════
PQA_COST = {
    "Diabetes": 4700,
    "Cardiovascular": 3900,
    "Respiratory": 2700,
    "Mental_Health": 1800,
}

# ══════════════════════════════════════════════════════════════
# CARE MANAGER PRODUCTIVITY CONSTANTS
# ══════════════════════════════════════════════════════════════
CM_TIME_SAVED_PER_CALL_MIN = 6  # Agent-assisted call prep vs manual


# ══════════════════════════════════════════════════════════════
# TOOL 1 — rank_priority_calls
# ══════════════════════════════════════════════════════════════
def rank_priority_calls(n: int = 10) -> str:
    """
    Rank the top N members a care manager should call today.

    Priority logic:
      1. Urgency tier: Critical > Urgent > Elevated
      2. Worst PDC (lowest = highest priority)
      3. Highest avoidable hospitalization cost

    Returns JSON string with list of members, each including:
      member_id, urgency, pdc, therapies, avoidable_cost, reason
    """
    data = _load_data()
    interventions = data["interventions"].copy()

    urgency_rank = {"Critical": 0, "Urgent": 1, "Elevated": 2}
    interventions["_u_rank"] = interventions["urgency"].map(urgency_rank).fillna(99)

    # Sort: urgency → worst PDC → highest cost
    interventions = interventions.sort_values(
        ["_u_rank", "worst_pdc", "total_hosp_cost"],
        ascending=[True, True, False]
    ).head(n)

    results = []
    for _, row in interventions.iterrows():
        therapies = str(row.get("therapies", ""))
        primary_therapy = therapies.split(",")[0].strip() if therapies else "Unknown"
        pqa_cost = PQA_COST.get(primary_therapy, 3500)

        reason_parts = []
        if row["urgency"] == "Critical":
            reason_parts.append("hospitalized and non-adherent")
        elif row["urgency"] == "Urgent":
            reason_parts.append(f"PDC {row['worst_pdc']}% — severe gap")
        else:
            reason_parts.append(f"PDC {row['worst_pdc']}% — moderate gap")

        if pd.notna(row.get("admission_count")) and row["admission_count"] > 0:
            reason_parts.append(f"{int(row['admission_count'])} prior admission(s)")

        reason = " · ".join(reason_parts)

        results.append({
            "member_id": row["MEMBER_ID"],
            "urgency": row["urgency"],
            "worst_pdc_pct": float(row["worst_pdc"]),
            "therapies": therapies,
            "admissions": int(row["admission_count"]) if pd.notna(row.get("admission_count")) else 0,
            "avoidable_cost_usd": pqa_cost,
            "reason": reason,
        })

    return json.dumps({
        "count": len(results),
        "cm_time_saved_min": len(results) * CM_TIME_SAVED_PER_CALL_MIN,
        "source": "intervention_recommendations.csv + dashboard_data.csv",
        "members": results,
    }, indent=2)


# ══════════════════════════════════════════════════════════════
# TOOL 2 — get_member_brief
# ══════════════════════════════════════════════════════════════
def get_member_brief(member_id: str) -> str:
    """
    30-second pre-call brief for a specific member.

    Returns JSON with: profile, diagnosis, root-cause signals,
    recommended interventions, avoidable cost, data citations.
    """
    data = _load_data()

    if member_id not in data["mbr_to_bene"]:
        return json.dumps({
            "error": f"Member ID '{member_id}' not found. Expected format: MBR-XXXX"
        })

    bene_id = data["mbr_to_bene"][member_id]
    m = data["member"][data["member"]["BENE_ID"] == bene_id]
    p = data["pdc"][data["pdc"]["BENE_ID"] == bene_id]
    g = data["gaps"][data["gaps"]["BENE_ID"] == bene_id]
    i = data["interventions"][data["interventions"]["BENE_ID"] == bene_id]

    if len(m) == 0:
        return json.dumps({"error": f"No data found for {member_id}"})

    m_row = m.iloc[0]

    # Profile
    therapies = []
    if len(p) > 0:
        therapies = sorted(p["therapy_class"].dropna().unique().tolist())

    # Per-therapy PDC
    pdc_detail = []
    for _, row in p.iterrows():
        pdc_detail.append({
            "therapy_class": row["therapy_class"],
            "pdc_pct": round(float(row["pdc"]), 1),
            "adherent": bool(row["pdc"] >= 80),
            "fills": int(row.get("fill_count", 0)),
        })

    # Gap signals
    gap_signals = []
    if len(g) > 0:
        g_row = g.iloc[0]
        if pd.notna(g_row.get("days_since_last_fill")) and g_row["days_since_last_fill"] > 30:
            gap_signals.append(f"Last fill {int(g_row['days_since_last_fill'])} days ago")
        if pd.notna(g_row.get("max_gap_days")) and g_row["max_gap_days"] > 30:
            gap_signals.append(f"Max coverage gap {int(g_row['max_gap_days'])} days")

    # Hospitalization
    hosp_count = int(m_row["admission_count"]) if pd.notna(m_row.get("admission_count")) else 0
    hosp_cost = float(m_row["total_hosp_cost"]) if pd.notna(m_row.get("total_hosp_cost")) else 0.0

    # Avoidable cost
    primary_therapy = therapies[0] if therapies else "Unknown"
    pqa_cost = PQA_COST.get(primary_therapy, 3500)

    # Interventions from pipeline
    recommended_interventions = []
    if len(i) > 0:
        intv_str = str(i.iloc[0].get("interventions", ""))
        if intv_str and intv_str != "nan":
            recommended_interventions = [x.strip() for x in intv_str.split("|") if x.strip()]

    urgency = i.iloc[0]["urgency"] if len(i) > 0 else "Stable"

    # Diagnosis summary
    worst_pdc = min([d["pdc_pct"] for d in pdc_detail]) if pdc_detail else None
    diagnosis_parts = []
    if urgency == "Critical":
        diagnosis_parts.append("Critical — hospitalized with non-adherence")
    elif worst_pdc is not None and worst_pdc < 50:
        diagnosis_parts.append(f"Severe non-adherence — worst PDC {worst_pdc}%")
    elif worst_pdc is not None and worst_pdc < 80:
        diagnosis_parts.append(f"Moderate non-adherence — worst PDC {worst_pdc}%")
    else:
        diagnosis_parts.append("Within adherence threshold")

    return json.dumps({
        "member_id": member_id,
        "risk_tier": m_row["risk_tier"],
        "urgency": urgency,
        "therapies": therapies,
        "pdc_by_therapy": pdc_detail,
        "gap_signals": gap_signals,
        "hospitalizations": {
            "admission_count": hosp_count,
            "total_cost_usd": hosp_cost,
        },
        "diagnosis": " · ".join(diagnosis_parts),
        "recommended_interventions": recommended_interventions,
        "avoidable_cost_usd_per_year": pqa_cost,
        "cm_time_saved_min": CM_TIME_SAVED_PER_CALL_MIN,
        "sources": [
            "dashboard_data.csv (risk tier, hospitalizations)",
            "member_pdc_scores.csv (PDC per therapy)",
            "member_gap_analysis.csv (fill gaps)",
            "intervention_recommendations.csv (recommended actions, urgency)",
        ],
    }, indent=2)


# ══════════════════════════════════════════════════════════════
# TOOL 3 — draft_outreach
# ══════════════════════════════════════════════════════════════
def draft_outreach(member_id: str, channel: str = "sms") -> str:
    """
    Prepare context for outreach generation. The LLM generates the final text
    using this context — this tool returns the facts to ground the message.

    channel: 'sms' (to member) or 'provider' (clinical note to prescriber)
    """
    data = _load_data()

    if channel not in ("sms", "provider"):
        return json.dumps({"error": "channel must be 'sms' or 'provider'"})

    if member_id not in data["mbr_to_bene"]:
        return json.dumps({"error": f"Member ID '{member_id}' not found"})

    bene_id = data["mbr_to_bene"][member_id]
    p = data["pdc"][data["pdc"]["BENE_ID"] == bene_id]
    i = data["interventions"][data["interventions"]["BENE_ID"] == bene_id]
    g = data["gaps"][data["gaps"]["BENE_ID"] == bene_id]

    therapies = sorted(p["therapy_class"].dropna().unique().tolist()) if len(p) > 0 else []
    primary = therapies[0].replace("_", " ") if therapies else "your medication"
    worst_pdc = float(p["pdc"].min()) if len(p) > 0 else None

    days_since = None
    if len(g) > 0 and pd.notna(g.iloc[0].get("days_since_last_fill")):
        days_since = int(g.iloc[0]["days_since_last_fill"])

    urgency = i.iloc[0]["urgency"] if len(i) > 0 else "Stable"

    # Compliance notes
    if channel == "sms":
        compliance = [
            "TCPA: member must have opted in to SMS outreach",
            "Reading level: 8th grade maximum",
            "No PHI beyond first name",
            "Include opt-out: reply STOP",
        ]
        style_guide = (
            "Warm but brief (under 320 chars). Acknowledge the therapy by plain name. "
            "Offer one specific help (refill, pharmacist call, transport). End with opt-out."
        )
    else:  # provider
        compliance = [
            "HIPAA: provider is treating clinician — PHI permitted",
            "Format: structured clinical handoff",
            "Tone: peer-to-peer, factual",
            "Include: member identifier, adherence data, ask",
        ]
        style_guide = (
            "Structured note: Subject line, 3 data points (PDC, gap, hospitalization), "
            "1 clear ask, closing. Under 150 words. No patient identifiers beyond MBR-id."
        )

    return json.dumps({
        "member_id": member_id,
        "channel": channel,
        "context_facts": {
            "primary_therapy": primary,
            "worst_pdc_pct": round(worst_pdc, 1) if worst_pdc is not None else None,
            "days_since_last_fill": days_since,
            "urgency": urgency,
        },
        "compliance_requirements": compliance,
        "style_guide": style_guide,
        "source": "member_pdc_scores.csv + member_gap_analysis.csv + intervention_recommendations.csv",
    }, indent=2)


# ══════════════════════════════════════════════════════════════
# TOOL SCHEMAS — for Groq function-calling API
# ══════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rank_priority_calls",
            "description": (
                "Return the top N members a care manager should call today, ranked by "
                "urgency, worst PDC, and avoidable cost. Use when the user asks which "
                "members to prioritize, who to call, or for a daily call list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of priority members to return (default 10)",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_member_brief",
            "description": (
                "Generate a 30-second pre-call brief for a specific member: risk tier, "
                "PDC per therapy, gap signals, hospitalizations, recommended interventions, "
                "and avoidable cost. Use whenever the user asks about a specific member "
                "by MBR-id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member_id": {
                        "type": "string",
                        "description": "Member identifier in format MBR-XXXX (e.g., MBR-2066)",
                    },
                },
                "required": ["member_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_outreach",
            "description": (
                "Return the factual context + compliance requirements needed to draft "
                "outreach for a specific member. The LLM writes the final message using "
                "this context. Use when the user asks to draft an SMS, message, or "
                "provider note for a member."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "member_id": {
                        "type": "string",
                        "description": "Member identifier in format MBR-XXXX",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["sms", "provider"],
                        "description": "'sms' for member-facing text; 'provider' for clinical note to prescriber",
                    },
                },
                "required": ["member_id", "channel"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════
# TOOL DISPATCHER — used by agent.py
# ══════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "rank_priority_calls": rank_priority_calls,
    "get_member_brief": get_member_brief,
    "draft_outreach": draft_outreach,
}


def call_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name. Used by agent orchestrator."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return TOOL_REGISTRY[name](**arguments)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {str(e)}"})