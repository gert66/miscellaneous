# Install dependencies:
#   pip install streamlit requests anthropic
#
# Set your API key:
#   export ANTHROPIC_API_KEY=your-key-here
#
# Run the app:
#   streamlit run company_scraper.py

import json
import re

import anthropic
import requests
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a company research assistant. Extract information from the provided "
    "website text. Return ONLY a raw JSON object with no markdown and no backticks, "
    "using this exact structure:\n"
    '{"founder": {"value": "...", "source_text": "..."},\n'
    ' "description": {"value": "...", "source_text": "..."},\n'
    ' "address": {"value": "...", "source_text": "..."},\n'
    ' "phone": {"value": "...", "source_text": "..."},\n'
    ' "email": {"value": "...", "source_text": "..."}}\n'
    "Set both sub-fields to null for any field you cannot find. "
    "source_text should be the exact sentence or phrase where you found the value."
)

JINA_BASE = "https://r.jina.ai/"
MAX_CONTENT_CHARS = 40_000

FIELDS = ["founder", "description", "address", "phone", "email"]
FIELD_LABELS = {
    "founder": "Founder",
    "description": "Description",
    "address": "Address",
    "phone": "Phone",
    "email": "Email",
}


# ── Jina fetching ─────────────────────────────────────────────────────────────

def fetch_jina(url: str) -> str | None:
    """Fetch clean readable text for a URL via Jina Reader."""
    try:
        resp = requests.get(JINA_BASE + url, timeout=20)
        resp.raise_for_status()
        return resp.text[:MAX_CONTENT_CHARS]
    except Exception:
        return None


# ── Claude extraction ─────────────────────────────────────────────────────────

def extract_company_info(url: str) -> tuple[dict, anthropic.types.Usage]:
    """
    Fetch main page and /about-us via Jina, send combined text to Claude,
    return parsed info dict and API usage.
    """
    main_text = fetch_jina(url) or ""
    about_text = fetch_jina(url.rstrip("/") + "/about-us") or ""

    combined = ""
    if main_text:
        combined += f"=== Content from {url} ===\n{main_text}\n\n"
    if about_text:
        combined += f"=== Content from {url}/about-us ===\n{about_text}\n"

    if not combined:
        raise RuntimeError("Could not fetch any content from the provided URL.")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Extract the founder name, description, address, phone, and email "
                    f"from the following website content:\n\n{combined}"
                ),
            }
        ],
    )

    result = _parse_json_from_response(response.content)
    if result is not None:
        return result, response.usage
    raise json.JSONDecodeError("No JSON object found in Claude's response", "", 0)


def _parse_json_from_response(content: list) -> dict | None:
    """Search all text blocks for the first valid JSON object."""
    for block in content:
        if not (hasattr(block, "type") and block.type == "text"):
            continue
        obj_match = re.search(r"\{.*\}", block.text, re.DOTALL)
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
    source_text = entry.get("source_text") or ""

    col1, col2 = st.columns([3, 1])
    with col1:
        if value:
            st.markdown(f"**{label}:** {value}")
        else:
            st.markdown(f"**{label}:** *(not found)*")
    with col2:
        with st.expander("Source"):
            if source_text:
                st.code(source_text, language=None)
            else:
                st.markdown("*(no source)*")


# ── App ───────────────────────────────────────────────────────────────────────

st.title("Company Info Scraper")
st.write(
    "Enter a company website URL. The page content is fetched via Jina Reader "
    "and sent to Claude to extract structured company information."
)

url = st.text_input("Company Website URL", placeholder="https://example.com")

if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            with st.spinner("Fetching content via Jina and extracting with Claude..."):
                info, usage = extract_company_info(url)

            st.success("Done!")

            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)
            st.caption(
                f"🔢 Tokens gebruikt: {input_tokens} input / {output_tokens} output"
                f" | Geschatte kosten: ${cost:.4f}"
            )

            st.header("Extracted Company Information")
            for field in FIELDS:
                entry = info.get(field) or {"value": None, "source_text": None}
                render_field(FIELD_LABELS[field], entry)

            if all(not (info.get(f) or {}).get("value") for f in FIELDS):
                st.info(
                    "Claude could not find any of the requested fields. "
                    "The site may block Jina Reader or not publish this information."
                )

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
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
