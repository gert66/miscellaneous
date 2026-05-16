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
    "You are a company research assistant. Use web search to find information "
    "about the company at the given URL. Extract: founder name, company description, "
    "address, phone number, and email address. "
    "Return ONLY a JSON object with this exact structure — no prose, no markdown fences:\n"
    '{"founder": {"value": "...", "source_url": "..."},\n'
    ' "description": {"value": "...", "source_url": "..."},\n'
    ' "address": {"value": "...", "source_url": "..."},\n'
    ' "phone": {"value": "...", "source_url": "..."},\n'
    ' "email": {"value": "...", "source_url": "..."}}\n'
    "Set value to null for any field you cannot find. "
    "source_url should be the page where you found the information."
)

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

FIELDS = ["founder", "description", "address", "phone", "email"]
FIELD_LABELS = {
    "founder": "Founder",
    "description": "Description",
    "address": "Address",
    "phone": "Phone",
    "email": "Email",
}


# ── Claude search + extraction ────────────────────────────────────────────────

def search_company_info(url: str) -> dict:
    """
    Single-turn API call with web_search enabled. Claude searches autonomously
    and returns a text response; we extract the JSON from the text blocks.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[WEB_SEARCH_TOOL],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research the company at {url}. Use web search to explore the website "
                    "thoroughly — including any about, team, or contact pages you find. "
                    "Find the founder or key person behind the company, a description of what "
                    "they do, their address, phone, and email. Return ONLY a JSON object with "
                    "fields: founder, description, address, phone, email, and for each a "
                    "source_text field showing where you found it."
                ),
            }
        ],
    )

    result = _parse_json_from_response(response.content)
    if result is not None:
        return result
    raise json.JSONDecodeError("No JSON object found in Claude's response", "", 0)


def _parse_json_from_response(content: list) -> dict | None:
    """Search all text blocks for the first valid JSON object."""
    for block in content:
        if not (hasattr(block, "type") and block.type == "text"):
            continue
        text = block.text
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not obj_match:
            continue
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            continue
    return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def render_field(label: str, entry: dict) -> None:
    value = entry.get("value")
    source_url = entry.get("source_url") or ""

    col1, col2 = st.columns([3, 1])
    with col1:
        if value:
            st.markdown(f"**{label}:** {value}")
        else:
            st.markdown(f"**{label}:** *(not found)*")
    with col2:
        with st.expander("Source"):
            if source_url:
                st.markdown(f"[{source_url}]({source_url})")
            else:
                st.markdown("*(no source)*")


# ── App ───────────────────────────────────────────────────────────────────────

st.title("Company Info Scraper")
st.write(
    "Enter a company website URL. Claude will use web search to find the "
    "founder, description, address, phone, and email."
)

url = st.text_input("Company Website URL", placeholder="https://example.com")

if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            with st.spinner("Searching with Claude..."):
                info = search_company_info(url)

            st.success("Done!")

            st.header("Extracted Company Information")
            for field in FIELDS:
                entry = info.get(field) or {"value": None, "source_url": None}
                render_field(FIELD_LABELS[field], entry)

            if all(not (info.get(f) or {}).get("value") for f in FIELDS):
                st.info("Claude could not find any of the requested fields for this company.")

            with st.expander("Raw Claude JSON (debug)"):
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
