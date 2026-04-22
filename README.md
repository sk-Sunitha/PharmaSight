# PharmaSight Agent

**A pharmacy-informed AI agent for chronic disease care management.**

Live demo: [pharmasight-agent.streamlit.app](https://pharmasight-agent.streamlit.app/)

---

## What it is

PharmaSight Agent is an end-to-end AI product that turns raw pharmacy claims data into a prioritized action list for care managers — and uses a grounded LLM agent to compress the work of a care-management call from ten minutes to four.

It does three things that fit together:

1. **Data pipeline** — ingests raw pharmacy claims, classifies medications into therapy classes, calculates Proportion of Days Covered (PDC), tiers members by risk, and estimates avoidable hospitalization cost.
2. **Analytics dashboard** — four views that answer the questions a clinical program manager actually asks: how big is the adherence problem, where is it concentrated, what does it cost, and who should we contact first.
3. **AI agent layer** — a grounded copilot with three tool-calling capabilities: priority call ranking, pre-call member briefs, and compliant outreach drafting. Not a chatbot — every output traces to a specific data row.

---

## Why this matters

Pharmacy non-adherence is one of the largest, most measurable, and most addressable cost drivers in population health. Published research estimates $100B–$300B in annual avoidable healthcare spend in the US tied to medication non-adherence. Care managers — pharmacists and nurses paid to close this gap — spend roughly 40% of their day on context-gathering, not on actual outreach.

The opportunity: if you can compress the context-gathering step and ground AI recommendations in real claims data, you unlock both clinical outcomes and operational leverage simultaneously.

This project demonstrates that I can identify that opportunity, build the pipeline that produces the signal, design the AI layer that acts on it, and quantify the ROI for a buyer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STREAMLIT UI — 5 tabs                                      │
│  Population Overview · Adherence Deep Dive ·                │
│  Risk & Cost Impact · Intervention Queue · Agent            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT ORCHESTRATOR (agent.py)                              │
│  Groq API · Llama-3.3-70B · OpenAI-compatible function      │
│  calling · tool-calling loop with grounded citations        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  TOOLS LAYER (tools.py)                                     │
│  rank_priority_calls() · get_member_brief() ·               │
│  draft_outreach()                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (steps 1–7)                                  │
│  Raw claims → NDC classification (RxNorm/ATC APIs) →        │
│  Therapy-class enrichment → PDC calculation →               │
│  Risk tiering → Intervention mapping → Dashboard data       │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle:** the system is data-source agnostic. The input is official synthetic healthcare claims published by CMS. Swap the input for real claims from any plan, and the same pipeline and agent produce the same structured outputs — the numbers change, the system doesn't.

---

## Data source

Official synthetic claims data published by CMS (Centers for Medicare & Medicaid Services). The files used are public, research-ready, and contain no PHI. Specifically:

| File | What it contains |
|---|---|
| Beneficiary Summary | Member demographics, enrollment, chronic condition indicators |
| Prescription Drug Event | Every pharmacy fill — NDC, date, days supplied, cost |
| Inpatient | Every hospital admission — diagnosis, dates, claim payment amount |

Synthetic data means the numbers are directional, not production-accurate. The methodology, pipeline, and agent design are production-grade.

---

## The analytics pipeline

Seven stages, each a standalone script so the pipeline is reproducible and auditable.

1. **NDC verification** — confirm prescription NDCs resolve to real drugs via RxNorm API (90% hit rate)
2. **NDC-to-therapy-class mapping** — map each NDC to ATC therapy class using the RxClass API (parallelized to hit 22 requests/sec)
3. **Chronic member identification** — filter to members with at least one fill in the four target therapy classes: Cardiovascular, Diabetes, Respiratory, Mental Health
4. **PDC calculation** — compute Proportion of Days Covered per member per therapy class using day-by-day coverage math, following the CMS measure specification
5. **Risk tiering** — assign each member to High / Medium / Low risk based on PDC + hospitalization history
6. **Intervention recommendation** — match each non-adherent member to PQA-approved interventions based on their gap pattern, therapy mix, and risk tier
7. **Dashboard data build** — denormalize everything into the four CSVs the dashboard and agent consume

Numbers at the end: 2,075 chronic members, 2,799 PDC scores, 1,256-member intervention queue.

---

## The dashboard

Four views, each answering one executive-level question.

| View | The question it answers |
|---|---|
| **Population Overview** | How big is the problem? How is risk distributed, and does it correlate with hospitalization? |
| **Adherence Deep Dive** | Which therapy class has the worst adherence, and how does the PDC distribution look? |
| **Risk & Cost Impact** | How much of the hospitalization spend is attached to non-adherent members? What's avoidable? |
| **Intervention Queue** | Exactly which members need outreach this week, ranked by urgency, with recommended actions? |

Design choices: bordered card containers for each chart, subtle color palette (no red-green-yellow glare), direct labels on bars and pie slices, dashed reference line for the 80% CMS adherence threshold, tooltips for deeper data on hover.

---

## The agent layer

Three tool-calling capabilities, each mapping to a moment in a care manager's day.

| Capability | Workflow moment | What the agent does |
|---|---|---|
| **Today's Priority Calls** | Before the day starts | Ranks the top N members by urgency + worst PDC + avoidable cost, with a one-line reason for each |
| **Member Brief** | Before a specific call | 30-second grounded brief: PDC per therapy, gap signals, hospitalizations, PQA-approved interventions, avoidable cost |
| **Outreach Drafter** | After the call | Compliance-aware draft: TCPA-compliant SMS to member, or structured clinical handoff to provider |

**Why this qualifies as an agent, not a chatbot:**

| Criterion | Implementation |
|---|---|
| Tool use | Calls one of three Python functions — `rank_priority_calls`, `get_member_brief`, `draft_outreach` |
| Data grounding | Never answers from LLM training data; must call a tool and cite its output |
| Multi-step reasoning | System prompt enforces structured output: diagnosis → root cause → actions → avoidable cost → sources |
| Transparency | Every response is followed by a tool-trace expander showing exactly which tool was called, with what arguments, and what it returned |

Model: Llama-3.3-70B via Groq (free tier, OpenAI-compatible function calling, 400 tokens/sec).

---

## ROI framing

The system addresses value in three layers — one buyer captures all three.

**Layer 1 — Clinical and financial ROI**
Avoidable hospitalization cost for non-adherent members in the cohort, calculated using PQA literature norms ($4,700/year Diabetes · $3,900 Cardiovascular · $2,700 Respiratory · $1,800 Mental Health). For this 2,075-member panel the addressable pool is ~$17.5M.

**Layer 2 — Operational ROI**
Agent-assisted call prep saves ~6 minutes per call. A care manager doing 20 calls/day reclaims ~2 hours of capacity — roughly 25% of a fully loaded RN/pharmacist cost. Same headcount, 25% more coverage.

**Layer 3 — Compliance and quality ROI**
Better adherence flows directly into CMS quality measures. For plans operating under Star Ratings, triple-weighted adherence measures feed into bonus payments. The connection from pipeline output to quality bonus is mechanical, not speculative.

Together these create a single pitch: one tool, three P&L lines, one buyer.

---

## What this project is trying to prove

This is a portfolio project, not a production system. The goal is to show that I can:

- **See product opportunity** in messy, publicly available healthcare data
- **Build the pipeline** that turns raw claims into decision-grade signal
- **Design the AI layer** that acts on the signal — with grounding, tool use, and transparent reasoning
- **Frame the ROI** in the language the buyer (health plan executive) speaks

None of the individual pieces are novel. The combination — pipeline + dashboard + agent, all in one coherent product story — is the point.

---

## What I'd build next

Keeping v1 deliberately narrow. Clear v2 extensions:

- **RAG over formularies and CMS measure specs** — let the agent reason over plan-specific drug lists and quality measure definitions
- **Feedback loop** — care manager rates each agent output; use signal to improve prompts or fine-tune a smaller model
- **Transcript intake** — paste a call recording, get an auto-drafted SOAP note and follow-up plan
- **A/B test designer** — the agent proposes intervention experiments (outreach cadence, channel mix) and sizes expected lift
- **Live claims integration** — swap synthetic data for a plan's live data feed; same pipeline, real numbers

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Pipeline | pandas, RxNorm API, RxClass API (ATC mapping) |
| Dashboard | Streamlit, Altair |
| Agent | Groq (Llama-3.3-70B, OpenAI-compatible function calling) |
| Hosting | Streamlit Community Cloud |
| Source control | GitHub |

---

## Repo structure

```
PharmaSight/
├── app.py                         # Streamlit dashboard + Agent tab
├── agent.py                       # Orchestrator — Groq + tool-calling loop
├── tools.py                       # The 3 tools the agent can call
├── requirements.txt               # Python dependencies
│
├── step1_verify_ndc.py            # Pipeline: NDC verification via RxNorm
├── step2_v2_parallel.py           # Pipeline: NDC → ATC therapy class (parallelized)
├── step3_identify_chronic_members.py  # Pipeline: chronic cohort filter
├── step4_pdc_real.py              # Pipeline: day-by-day PDC calculation
├── step5_risk_cost.py             # Pipeline: risk tiering + cost attribution
├── step6_interventions.py         # Pipeline: PQA-based intervention mapping
├── step7_build_dashboard_data.py  # Pipeline: final denormalized CSVs
│
├── dashboard_data.csv             # Member-level dashboard feed
├── member_pdc_scores.csv          # PDC per member per therapy class
├── intervention_recommendations.csv  # Prioritized intervention queue
└── member_gap_analysis.csv        # Fill-gap signals per member
```

---

## Running locally

```bash
git clone https://github.com/sk-sunitha/PharmaSight.git
cd PharmaSight
pip install -r requirements.txt

# Set the Groq key — get one free at console.groq.com
export GROQ_API_KEY="gsk_..."

streamlit run app.py
```

For the deployed version, `GROQ_API_KEY` is stored in Streamlit Secrets.

---

## About

Built as an AI Product Management portfolio project — demonstrating end-to-end ownership from raw data ingestion through product framing.

Repository: [github.com/sk-sunitha/PharmaSight](https://github.com/sk-sunitha/PharmaSight)
