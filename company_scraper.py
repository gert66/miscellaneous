# Install dependencies:
#   pip install streamlit anthropic
#
# Set your API key:
#   export ANTHROPIC_API_KEY=your-key-here
#
# Run the app:
#   streamlit run company_scraper.py

import json
import re

import anthropic
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a company data extraction assistant, similar to Lusha. "
    "Use web search to research the given company and extract structured data. "
    "Return ONLY a raw JSON object — no markdown, no backticks, no explanation — "
    "with exactly these fields:\n"
    '{\n'
    '  "name": "Full legal company name",\n'
    '  "description": "What the company does (1-2 sentences)",\n'
    '  "domain": "Primary website domain, e.g. example.com",\n'
    '  "employees": "Employee count range, e.g. 11-50",\n'
    '  "founded": "Year founded, e.g. 2012",\n'
    '  "mainIndustry": "Top-level industry, e.g. Technology",\n'
    '  "subIndustry": "More specific category, e.g. SaaS / HR Tech",\n'
    '  "companyType": "One of: Public Company, Private, Startup, Non-Profit, Government",\n'
    '  "address": "Full headquarters street address if available",\n'
    '  "location": {"city": "...", "state": "...", "country": "..."},\n'
    '  "website": "Full URL, e.g. https://example.com",\n'
    '  "linkedin": "LinkedIn company page URL or null"\n'
    '}\n'
    "Set any unknown field to null."
)

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

FIELDS = [
    ("name",        "Company Name"),
    ("description", "Description"),
    ("domain",      "Domain"),
    ("website",     "Website"),
    ("linkedin",    "LinkedIn"),
    ("employees",   "Employees"),
    ("founded",     "Founded"),
    ("mainIndustry","Industry"),
    ("subIndustry", "Sub-Industry"),
    ("companyType", "Company Type"),
    ("address",     "Address"),
    ("location",    "Location"),
]


# ── Claude extraction ─────────────────────────────────────────────────────────

def lookup_company(query: str) -> tuple[dict, anthropic.types.Usage]:
    """
    Single-turn Claude call with web_search. Returns (parsed dict, usage).
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        tools=[WEB_SEARCH_TOOL],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Look up the company at or associated with: {query}\n"
                    "Search for their official website, LinkedIn page, Crunchbase profile, "
                    "or any reliable source to find: company name, description, industry, "
                    "employee count, founding year, headquarters location, company type, "
                    "and LinkedIn URL. Return ONLY the JSON object."
                ),
            }
        ],
    )

    result = _parse_json(response.content)
    if result is not None:
        return result, response.usage
    raise json.JSONDecodeError("No JSON object found in Claude's response", "", 0)


def _parse_json(content: list) -> dict | None:
    """Search all text blocks for the first valid JSON object."""
    for block in content:
        if not (hasattr(block, "type") and block.type == "text"):
            continue
        match = re.search(r"\{.*\}", block.text, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def _fmt_location(loc) -> str:
    if not isinstance(loc, dict):
        return str(loc) if loc else "—"
    parts = [loc.get("city"), loc.get("state"), loc.get("country")]
    return ", ".join(p for p in parts if p) or "—"


def render_results(info: dict) -> None:
    st.header("Company Profile")

    # Hero row: name + type badge
    name = info.get("name") or "Unknown"
    company_type = info.get("companyType") or ""
    st.subheader(name + (f"  `{company_type}`" if company_type else ""))

    desc = info.get("description")
    if desc:
        st.write(desc)

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Domain**")
        st.write(info.get("domain") or "—")

        st.markdown("**Website**")
        website = info.get("website")
        st.write(f"[{website}]({website})" if website else "—")

        st.markdown("**LinkedIn**")
        linkedin = info.get("linkedin")
        st.write(f"[View profile]({linkedin})" if linkedin else "—")

        st.markdown("**Founded**")
        st.write(info.get("founded") or "—")

    with col_b:
        st.markdown("**Employees**")
        st.write(info.get("employees") or "—")

        st.markdown("**Industry**")
        main = info.get("mainIndustry") or "—"
        sub = info.get("subIndustry")
        st.write(f"{main} › {sub}" if sub else main)

        st.markdown("**Location**")
        st.write(_fmt_location(info.get("location")))

        st.markdown("**Address**")
        st.write(info.get("address") or "—")


# ── App ───────────────────────────────────────────────────────────────────────

st.title("Company Lookup")
st.write(
    "Enter a company website URL or domain name. "
    "Claude will use web search to extract Lusha-style company data."
)

query = st.text_input("Company URL or Domain", placeholder="example.com")

if st.button("Look Up Company"):
    if not query.strip():
        st.warning("Please enter a URL or domain.")
    else:
        try:
            with st.spinner("Searching with Claude..."):
                info, usage = lookup_company(query.strip())

            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)
            st.caption(
                f"🔢 Tokens gebruikt: {input_tokens} input / {output_tokens} output"
                f" | Geschatte kosten: ${cost:.4f}"
            )

            render_results(info)

            if all(not info.get(k) for k, _ in FIELDS):
                st.info("Claude could not find data for this company.")

            with st.expander("Raw JSON (debug)"):
                st.json(info)

        except json.JSONDecodeError:
            st.error(
                "Claude returned a response that could not be parsed as JSON. "
                "Try again — this can happen occasionally."
            )
        except anthropic.AuthenticationError:
            st.error(
                "Invalid Anthropic API key. "
                "Set the ANTHROPIC_API_KEY environment variable and restart the app."
            )
        except anthropic.APIStatusError as e:
            st.error(f"Claude API error ({e.status_code}): {e.message}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
