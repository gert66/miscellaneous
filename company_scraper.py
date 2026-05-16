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
from urllib.parse import urljoin, urlparse

import anthropic
import requests
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────

# Realistic browser headers so sites don't block the request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

MODEL = "claude-sonnet-4-20250514"

# Exact system prompt requested by the user
SYSTEM_PROMPT = (
    "You are a web scraping assistant. Extract key company information from this HTML. "
    "Return a JSON object with fields: founder, description, address, phone, email. "
    "For each field also include a source_text field showing the exact sentence from the "
    "HTML where you found it."
)

# Strip scripts/styles then cap HTML size before sending to Claude
MAX_HTML_CHARS = 80_000

FIELDS = ["founder", "description", "address", "phone", "email"]
FIELD_LABELS = {
    "founder": "Founder",
    "description": "Description",
    "address": "Address",
    "phone": "Phone",
    "email": "Email",
}


# ── HTML fetching ─────────────────────────────────────────────────────────────

def fetch_html(url: str) -> str | None:
    """Fetch a URL with browser headers. Returns HTML text or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def clean_html(html: str) -> str:
    """Remove script/style/comment noise and trim to MAX_HTML_CHARS."""
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    html = re.sub(r"\s{3,}", " ", html)
    return html[:MAX_HTML_CHARS]


def find_about_url(base_url: str, html: str) -> str | None:
    """Return the /about or /about-us URL if one exists, else None."""
    about_re = re.compile(r"/about(?:-us)?/?$", re.I)

    # Look for links already on the page
    for match in re.finditer(r'href=["\']([^"\']+)["\']', html, re.I):
        full = urljoin(base_url, match.group(1).strip())
        if about_re.search(urlparse(full).path):
            return full

    # Probe common paths directly
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    for candidate in ("/about", "/about-us"):
        try:
            r = requests.head(root + candidate, headers=HEADERS, timeout=6, allow_redirects=True)
            if r.status_code < 400:
                return root + candidate
        except Exception:
            pass
    return None


# ── Claude extraction ─────────────────────────────────────────────────────────

def extract_info(html: str, source_url: str) -> dict:
    """
    Send cleaned HTML to Claude and return a dict where each key maps to
    {"value": "...", "source_text": "..."}.
    """
    client = anthropic.Anthropic()

    user_message = (
        f"Here is the HTML from {source_url}.\n\n"
        "Return your answer as a JSON object. For each of the five fields "
        "(founder, description, address, phone, email) use this exact nested structure:\n"
        '  "<field>": {"value": "<extracted text>", "source_text": "<exact snippet from HTML>"}\n'
        "Set both sub-fields to null if the information is not present.\n\n"
        f"HTML:\n{html}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "")

    # Strip optional markdown code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else raw

    # Fall back to the first {...} block if the whole string isn't valid JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        obj_match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))
        raise


# ── Result merging ────────────────────────────────────────────────────────────

def merge_results(main: dict, main_url: str, about: dict, about_url: str | None) -> dict:
    """
    Merge extractions from two pages.  For each field, prefer the about-page
    value if it is non-null, otherwise fall back to the main-page value.
    Returns {field: {"value": ..., "source_text": ..., "source_url": ...}}.
    """
    merged: dict = {}
    for field in FIELDS:
        about_entry = about.get(field) if about else None
        main_entry = main.get(field)

        if _has_value(about_entry):
            merged[field] = {**about_entry, "source_url": about_url}
        elif _has_value(main_entry):
            merged[field] = {**main_entry, "source_url": main_url}
        else:
            merged[field] = {"value": None, "source_text": None, "source_url": main_url}
    return merged


def _has_value(entry: dict | None) -> bool:
    return bool(entry and entry.get("value"))


# ── UI helpers ────────────────────────────────────────────────────────────────

def render_field(label: str, entry: dict) -> None:
    """Render one extracted field plus its Source Evidence expander."""
    value = entry.get("value")
    source_text = entry.get("source_text")
    source_url = entry.get("source_url", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        if value:
            st.markdown(f"**{label}:** {value}")
        else:
            st.markdown(f"**{label}:** *(not found)*")
    with col2:
        with st.expander("Source Evidence"):
            st.markdown(f"**Source URL:** {source_url}")
            if source_text:
                st.markdown("**Exact text from page:**")
                st.code(source_text, language=None)
            else:
                st.markdown("*(no matching text found)*")


# ── App ───────────────────────────────────────────────────────────────────────

st.title("Company Info Scraper")
st.write(
    "Enter a company website URL. The page HTML is fetched with realistic browser headers, "
    "then sent to Claude to extract structured company information."
)

url = st.text_input("Company Website URL", placeholder="https://example.com")

if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            # 1. Fetch main page HTML
            with st.spinner("Fetching main page..."):
                main_html = fetch_html(url)
            if not main_html:
                st.error("Could not fetch the page. Check the URL and try again.")
                st.stop()

            # 2. Find and fetch the about page
            with st.spinner("Looking for About page..."):
                about_url = find_about_url(url, main_html)
                about_html = None
                if about_url and about_url.rstrip("/") != url.rstrip("/"):
                    about_html = fetch_html(about_url)

            # 3. Send to Claude for extraction
            with st.spinner("Sending main page to Claude..."):
                main_info = extract_info(clean_html(main_html), url)

            about_info: dict = {}
            if about_html:
                with st.spinner(f"Sending About page to Claude..."):
                    about_info = extract_info(clean_html(about_html), about_url)

            # 4. Merge results (prefer about-page values)
            merged = merge_results(main_info, url, about_info, about_url)

            st.success("Done!")

            # 5. Display
            st.header("Extracted Company Information")
            for field in FIELDS:
                render_field(FIELD_LABELS[field], merged[field])

            if all(not _has_value(merged[f]) for f in FIELDS):
                st.info(
                    "Claude could not find any of the requested fields on these pages. "
                    "The site may require JavaScript to render content, "
                    "or the information may not be publicly listed."
                )

            if about_url:
                st.caption(f"Also checked: {about_url}")
            else:
                st.caption("No /about or /about-us page found.")

            # Debug expander with raw Claude output
            with st.expander("Raw Claude JSON (debug)"):
                st.subheader("Main page extraction")
                st.json(main_info)
                if about_info:
                    st.subheader("About page extraction")
                    st.json(about_info)

        except json.JSONDecodeError:
            st.error(
                "Claude returned a response that could not be parsed as JSON. "
                "Try again — this can happen with unusually structured pages."
            )
        except anthropic.AuthenticationError:
            st.error(
                "Invalid Anthropic API key. "
                "Set the ANTHROPIC_API_KEY environment variable and restart the app."
            )
        except anthropic.APIStatusError as e:
            st.error(f"Claude API error ({e.status_code}): {e.message}")
        except requests.exceptions.MissingSchema:
            st.error("Invalid URL. Make sure it starts with http:// or https://")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the URL. Check the address and try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error fetching page: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
