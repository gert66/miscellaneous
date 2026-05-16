# Install dependencies:
#   pip install streamlit requests beautifulsoup4
#
# Run the app:
#   streamlit run company_scraper.py

import re
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup, Tag

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CompanyScraper/1.0)"}

PERSON_ROLE_RE = re.compile(
    r"\b(founder|co-founder|ceo|chief executive|president|cto|coo|cfo|"
    r"director|chairman|chairwoman|managing partner|partner|head of)\b",
    re.I,
)
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")


# ── Evidence helpers ─────────────────────────────────────────────────────────

def make_evidence(tag: Tag | None, source_url: str, highlight: str = "") -> dict:
    """Build a source-evidence dict from a BeautifulSoup tag."""
    if tag is None:
        return {"tag": "unknown", "context": "", "url": source_url}
    tag_name = tag.name or "unknown"
    context = tag.get_text(separator=" ", strip=True)
    # Trim very long contexts to 300 chars, keeping the highlight visible
    if len(context) > 300:
        if highlight:
            idx = context.lower().find(highlight.lower())
            start = max(0, idx - 80)
            context = ("..." if start > 0 else "") + context[start : start + 300] + "..."
        else:
            context = context[:300] + "..."
    attrs = {k: v for k, v in (tag.attrs or {}).items() if k in ("class", "id", "name", "itemtype")}
    attr_str = ""
    if attrs:
        parts = []
        for k, v in attrs.items():
            parts.append(f'{k}="{" ".join(v) if isinstance(v, list) else v}"')
        attr_str = " " + " ".join(parts)
    return {
        "tag": f"<{tag_name}{attr_str}>",
        "context": context,
        "url": source_url,
    }


def find_tag_containing(soup: BeautifulSoup, text: str) -> Tag | None:
    """Return the most specific tag whose text contains the given string."""
    for el in soup.find_all(string=re.compile(re.escape(text), re.I)):
        parent = el.parent
        if parent and parent.name not in ("script", "style"):
            return parent
    return None


def item(value: str, evidence: dict) -> dict:
    return {"value": value, "evidence": evidence}


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_contact(html_text: str, soup: BeautifulSoup, source_url: str) -> list[dict]:
    results = []
    seen_values: set[str] = set()

    def add(value: str, tag: Tag | None, highlight: str = ""):
        if value in seen_values:
            return
        seen_values.add(value)
        results.append(item(value, make_evidence(tag, source_url, highlight)))

    # Emails
    for email in re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", html_text):
        tag = find_tag_containing(soup, email)
        add(f"Email: {email}", tag, email)

    # Phone numbers
    for phone in re.findall(r"(?:\+?\d[\d\s\-().]{7,}\d)", html_text)[:5]:
        cleaned = phone.strip()
        if len(re.sub(r"\D", "", cleaned)) >= 7:
            tag = find_tag_containing(soup, cleaned[:10])
            add(f"Phone: {cleaned}", tag, cleaned[:10])

    # <address> tags
    for tag in soup.find_all("address"):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            add(f"Address: {text}", tag)

    # Divs/spans with address/location/contact in class or id
    addr_kw = re.compile(r"address|location|contact", re.I)
    for tag in soup.find_all(["div", "span", "p", "section"], class_=addr_kw):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) < 300:
            add(f"Contact block: {text}", tag)

    return results


def extract_people(soup: BeautifulSoup, source_url: str) -> list[dict]:
    results = []
    seen_names: set[str] = set()

    def add_person(name: str, role: str, tag: Tag | None):
        key = name.lower().strip()
        if key in seen_names or not name:
            return
        seen_names.add(key)
        label = name + (f" — {role}" if role else "")
        results.append(item(label, make_evidence(tag, source_url, name)))

    # schema.org Person
    for el in soup.find_all(attrs={"itemtype": re.compile(r"schema\.org/Person", re.I)}):
        name_el = el.find(attrs={"itemprop": "name"})
        role_el = el.find(attrs={"itemprop": re.compile(r"jobTitle|role", re.I)})
        name = name_el.get_text(strip=True) if name_el else ""
        role = role_el.get_text(strip=True) if role_el else ""
        if name:
            add_person(name, role, el)

    # Team card patterns
    card_kw = re.compile(r"team|member|person|people|founder|leadership|staff|bio", re.I)
    for card in soup.find_all(["div", "article", "li", "section"], class_=card_kw):
        heading = card.find(["h1", "h2", "h3", "h4", "h5", "strong", "b"])
        role_tag = card.find(["p", "span"], class_=re.compile(r"role|title|position|job", re.I))
        if heading:
            name = heading.get_text(strip=True)
            role = role_tag.get_text(strip=True) if role_tag else ""
            if NAME_RE.match(name) and len(name.split()) <= 5:
                add_person(name, role, card)

    # Keyword fallback
    if not results:
        for block in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "span"]):
            text = block.get_text(separator=" ", strip=True)
            if not PERSON_ROLE_RE.search(text):
                continue
            role_match = PERSON_ROLE_RE.search(text)
            role = role_match.group(0).title() if role_match else ""
            for name in NAME_RE.findall(text):
                if len(name.split()) >= 2:
                    add_person(name, role, block)

    return results


# ── Page fetching ────────────────────────────────────────────────────────────

def get_soup(url: str) -> tuple[requests.Response, BeautifulSoup] | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp, BeautifulSoup(resp.text, "html.parser")
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        return None


def scrape_page(url: str) -> dict:
    result = get_soup(url)
    if result is None:
        return {}
    resp, soup = result

    title = soup.title.string.strip() if soup.title and soup.title.string else "Not found"

    meta_desc = ""
    meta_tag = None
    for attrs in [{"name": "description"}, {"property": "og:description"}]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            meta_desc = tag["content"].strip()
            meta_tag = tag
            break

    meta_evidence = make_evidence(meta_tag, url) if meta_tag else None

    contact = extract_contact(resp.text, soup, url)
    people = extract_people(soup, url)

    return {
        "title": title,
        "meta_description": meta_desc or "Not found",
        "meta_evidence": meta_evidence,
        "contact_info": contact,
        "people": people,
        "url": url,
    }


def find_about_url(base_url: str, soup: BeautifulSoup) -> str | None:
    about_re = re.compile(r"/about(?:-us)?/?$", re.I)
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"].strip())
        if about_re.search(urlparse(full).path):
            return full
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    for candidate in ["/about", "/about-us"]:
        try:
            r = requests.head(root + candidate, headers=HEADERS, timeout=6, allow_redirects=True)
            if r.status_code < 400:
                return root + candidate
        except Exception:
            pass
    return None


def merge_items(a: list[dict], b: list[dict], key: str = "value") -> list[dict]:
    seen: set[str] = set()
    merged = []
    for it in a + b:
        k = it[key].lower().strip()
        if k not in seen:
            seen.add(k)
            merged.append(it)
    return merged


# ── UI helpers ───────────────────────────────────────────────────────────────

def render_evidence_expander(evidence: dict):
    with st.expander("Source Evidence"):
        st.markdown(f"**Source URL:** {evidence['url']}")
        st.markdown(f"**HTML element:** `{evidence['tag']}`")
        if evidence["context"]:
            st.markdown("**Surrounding text:**")
            st.code(evidence["context"], language=None)


def render_items(items: list[dict]):
    for it in items:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"- {it['value']}")
        with col2:
            render_evidence_expander(it["evidence"])


# ── App ──────────────────────────────────────────────────────────────────────

st.title("Company Info Scraper")
st.write("Enter a company website URL to extract title, description, contact info, and key people.")

url = st.text_input("Company Website URL", placeholder="https://example.com")

if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            with st.spinner("Scraping main page..."):
                main = scrape_page(url)
                if not main:
                    st.error("Could not fetch the page. Check the URL and try again.")
                    st.stop()

            about_data: dict = {}
            about_url: str | None = None
            with st.spinner("Looking for About page..."):
                result = get_soup(url)
                if result:
                    _, main_soup = result
                    about_url = find_about_url(url, main_soup)
                if about_url and about_url.rstrip("/") != url.rstrip("/"):
                    about_data = scrape_page(about_url)

            all_people = merge_items(main.get("people", []), about_data.get("people", []))
            all_contact = merge_items(main.get("contact_info", []), about_data.get("contact_info", []))

            st.success("Done!")

            # ── Main page ──
            st.header("Main Page")

            st.subheader("Page Title")
            st.write(main["title"])

            st.subheader("Meta Description")
            st.write(main["meta_description"])
            if main.get("meta_evidence"):
                render_evidence_expander(main["meta_evidence"])

            # ── About page ──
            if about_data:
                st.header(f"About Page")
                st.caption(about_url)

                st.subheader("Page Title")
                st.write(about_data.get("title", "Not found"))

                st.subheader("Meta Description")
                st.write(about_data.get("meta_description", "Not found"))
                if about_data.get("meta_evidence"):
                    render_evidence_expander(about_data["meta_evidence"])
            else:
                st.info("No /about or /about-us page found (or it returned an error).")

            # ── Key people ──
            st.header("Key People")
            if all_people:
                render_items(all_people)
            else:
                st.write("No key people found on these pages.")

            # ── Contact info ──
            st.header("Contact / Address Info")
            if all_contact:
                render_items(all_contact)
            else:
                st.write("No contact info found.")

        except requests.exceptions.MissingSchema:
            st.error("Invalid URL. Make sure it starts with http:// or https://")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the URL. Check the address and try again.")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
