"""
PharmaSight Agent — Orchestrator
Connects the 3 tools in tools.py to the Groq API via OpenAI-compatible
function calling. Implements a tool-calling loop: LLM picks a tool →
we execute it → feed result back → LLM writes grounded final answer.
"""

import json
import os
from groq import Groq
from tools import TOOL_SCHEMAS, call_tool


# ══════════════════════════════════════════════════════════════
# MODEL CONFIG
# ══════════════════════════════════════════════════════════════
# Llama-3.3 70B on Groq — free tier, fast, good at function calling
MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_HOPS = 4  # Safety: prevent infinite tool-calling loops


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT — the agent's job description
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are PharmaSight Agent — a grounded AI copilot for Medicare care managers
working on pharmacy adherence and chronic disease intervention.

RULES:
1. Always call a tool to get real data. Never invent facts, PDC values, or costs.
2. Cite data sources in every response. Use the "sources" or "source" field from tool output.
3. Use only PQA-approved intervention types (reference by name, e.g., "90-day refill conversion", "MTM consult", "refill reminder").
4. Keep tone clinical, concise, respectful. No hype language. No emojis.
5. If data is missing or ambiguous, say so explicitly. Never guess.

OUTPUT FORMAT for member briefs:
   Diagnosis:   [one line — what's wrong]
   Why:         [root cause from gap signals, hospitalization, PDC]
   Actions:     [2-3 bullets — PQA-approved interventions]
   Avoidable cost: $X/year (PQA norm for this therapy class)
   CM time saved: [minutes, from tool output]
   Sources: [list from tool output]

OUTPUT FORMAT for priority call lists:
   A table with Member · Urgency · PDC · Why it matters · Avoidable $
   Followed by one summary line on total CM time saved

OUTPUT FORMAT for outreach drafts:
   First echo the compliance requirements returned by the tool.
   Then produce the draft text inside a code block.
   - SMS: under 320 chars, 8th-grade reading level, include STOP opt-out
   - Provider note: structured clinical handoff, under 150 words
   Use member identifier (MBR-XXXX) — no other PHI.

If the user asks for something outside these three capabilities, say politely
that this version focuses on priority ranking, member briefs, and outreach drafting.
"""


# ══════════════════════════════════════════════════════════════
# CLIENT — reads key from environment (Streamlit Secrets injects it)
# ══════════════════════════════════════════════════════════════
def _get_client():
    """Create Groq client. Raises clear error if key missing."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. In Streamlit Cloud, add it in "
            "Settings → Secrets as: GROQ_API_KEY = \"gsk_...\""
        )
    return Groq(api_key=api_key)


# ══════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
# ══════════════════════════════════════════════════════════════
def run_agent(user_message: str, verbose: bool = False) -> dict:
    """
    Run the agent loop for a single user request.

    Returns dict with:
      - answer: final LLM response (string)
      - tool_calls: list of {tool, args, result} — shows what the agent did
      - error: error message if something failed (string or None)
    """
    try:
        client = _get_client()
    except RuntimeError as e:
        return {"answer": "", "tool_calls": [], "error": str(e)}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tool_trace = []

    # Tool-calling loop — at most MAX_TOOL_HOPS iterations
    for hop in range(MAX_TOOL_HOPS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1200,
            )
        except Exception as e:
            return {
                "answer": "",
                "tool_calls": tool_trace,
                "error": f"Groq API error: {str(e)}",
            }

        msg = response.choices[0].message

        # If LLM didn't request a tool, we have the final answer
        if not msg.tool_calls:
            return {
                "answer": msg.content or "",
                "tool_calls": tool_trace,
                "error": None,
            }

        # LLM wants to call one or more tools — execute them
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"[hop {hop}] calling {name}({args})")

            result = call_tool(name, args)
            tool_trace.append({"tool": name, "args": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Hit the hop limit — return whatever we have
    return {
        "answer": "Agent reached tool-call limit without finishing. This usually means the request was too complex.",
        "tool_calls": tool_trace,
        "error": "max_hops_exceeded",
    }


# ══════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPERS — called directly from the Streamlit buttons
# ══════════════════════════════════════════════════════════════
def get_priority_calls(n: int = 10) -> dict:
    """Capability D — Today's Priority Calls button."""
    return run_agent(
        f"Give me today's top {n} priority calls. For each member, show "
        f"their urgency, worst PDC, primary therapy, why they matter, and "
        f"avoidable cost. End with total CM time saved."
    )


def get_member_brief(member_id: str) -> dict:
    """Capability A — Member Brief button."""
    return run_agent(
        f"Generate a 30-second pre-call brief for member {member_id}. "
        f"Follow the standard brief output format."
    )


def get_outreach_draft(member_id: str, channel: str = "sms") -> dict:
    """Capability B — Outreach Drafter button."""
    channel_human = "SMS to the member" if channel == "sms" else "clinical note to the prescribing provider"
    return run_agent(
        f"Draft a {channel_human} for member {member_id}. "
        f"First echo the compliance requirements, then produce the draft text."
    )