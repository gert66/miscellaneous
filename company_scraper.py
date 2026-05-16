# Install dependencies:
#   pip install streamlit requests beautifulsoup4
#
# Run the app:
#   streamlit run company_scraper.py

import re
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CompanyScraper/1.0)"}

# Titles/roles that signal a key person mention
PERSON_ROLE_RE = re.compile(
    r"\b(founder|co-founder|ceo|chief executive|president|cto|coo|cfo|"
    r"director|chairman|chairwoman|managing partner|partner|head of)\b",
    re.I,
)

# Very rough heuristic: capitalised words that look like a name near a role keyword
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")


def get_soup(url: str) -> tuple[requests.Response, BeautifulSoup] | None:
    """Fetch URL and return (response, soup), or None on non-fatal failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp, BeautifulSoup(resp.text, "html.parser")
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        return None


def extract_contact(html_text: str, soup: BeautifulSoup) -> set[str]:
    contact = set()

    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", html_text)
    for e in emails:
        contact.add(f"Email: {e}")

    phones = re.findall(r"(?:\+?\d[\d\s\-().]{7,}\d)", html_text)
    for p in phones[:5]:
        cleaned = p.strip()
        if len(re.sub(r"\D", "", cleaned)) >= 7:
            contact.add(f"Phone: {cleaned}")

    for tag in soup.find_all("address"):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            contact.add(f"Address: {text}")

    addr_kw = re.compile(r"address|location|contact", re.I)
    for tag in soup.find_all(["div", "span", "p", "section"], class_=addr_kw):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) < 300:
            contact.add(f"Contact block: {text}")

    return contact


def extract_people(soup: BeautifulSoup) -> list[str]:
    """
    Look for person names near role keywords. Strategy:
    1. Search structured markup (schema.org Person, LinkedIn-style cards).
    2. Fall back to scanning paragraphs/headings for role keywords + nearby names.
    """
    found: dict[str, str] = {}  # name -> role

    # schema.org Person items
    for el in soup.find_all(attrs={"itemtype": re.compile(r"schema\.org/Person", re.I)}):
        name_el = el.find(attrs={"itemprop": "name"})
        role_el = el.find(attrs={"itemprop": re.compile(r"jobTitle|role", re.I)})
        name = name_el.get_text(strip=True) if name_el else ""
        role = role_el.get_text(strip=True) if role_el else ""
        if name:
            found[name] = role

    # Common team-card patterns: heading inside a card container
    card_kw = re.compile(r"team|member|person|people|founder|leadership|staff|bio", re.I)
    for card in soup.find_all(["div", "article", "li", "section"], class_=card_kw):
        heading = card.find(["h1", "h2", "h3", "h4", "h5", "strong", "b"])
        role_tag = card.find(["p", "span"], class_=re.compile(r"role|title|position|job", re.I))
        if heading:
            name = heading.get_text(strip=True)
            role = role_tag.get_text(strip=True) if role_tag else ""
            if name and NAME_RE.match(name) and len(name.split()) <= 5:
                found.setdefault(name, role)

    # Fallback: scan text blocks for role keywords and grab nearby capitalised names
    if not found:
        for block in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "span"]):
            text = block.get_text(separator=" ", strip=True)
            if not PERSON_ROLE_RE.search(text):
                continue
            names = NAME_RE.findall(text)
            role_match = PERSON_ROLE_RE.search(text)
            role = role_match.group(0).title() if role_match else ""
            for name in names:
                # Skip generic capitalized phrases
                if name.lower() in {"the", "our", "we", "they", "their", "this", "that"}:
                    continue
                if len(name.split()) >= 2:
                    found.setdefault(name, role)

    results = []
    for name, role in found.items():
        results.append(f"{name}" + (f" — {role}" if role else ""))
    return results


def scrape_page(url: str) -> dict:
    result = get_soup(url)
    if result is None:
        return {}
    resp, soup = result

    title = soup.title.string.strip() if soup.title and soup.title.string else "Not found"

    meta_desc = ""
    for attrs in [{"name": "description"}, {"property": "og:description"}]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            meta_desc = tag["content"].strip()
            break

    contact = extract_contact(resp.text, soup)
    people = extract_people(soup)

    return {
        "title": title,
        "meta_description": meta_desc or "Not found",
        "contact_info": sorted(contact) if contact else [],
        "people": people,
    }


def find_about_url(base_url: str, soup: BeautifulSoup) -> str | None:
    """Try to discover an /about or /about-us link from the page, else construct it."""
    about_re = re.compile(r"/about(?:-us)?/?$", re.I)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if about_re.search(urlparse(full).path):
            return full

    # Also try constructing the URLs directly
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    for candidate in ["/about", "/about-us"]:
        url = root + candidate
        try:
            r = requests.head(url, headers=HEADERS, timeout=6, allow_redirects=True)
            if r.status_code < 400:
                return url
        except Exception:
            pass
    return None


def merge_people(a: list[str], b: list[str]) -> list[str]:
    seen = set()
    merged = []
    for item in a + b:
        key = item.split(" — ")[0].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


# ── UI ──────────────────────────────────────────────────────────────────────

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

            # Try to find and scrape the about page
            about_data: dict = {}
            about_url: str | None = None
            with st.spinner("Looking for About page..."):
                result = get_soup(url)
                if result:
                    _, main_soup = result
                    about_url = find_about_url(url, main_soup)
                if about_url and about_url.rstrip("/") != url.rstrip("/"):
                    about_data = scrape_page(about_url)

            # Merge people from both pages
            all_people = merge_people(main.get("people", []), about_data.get("people", []))
            all_contact = sorted(
                set(main.get("contact_info", [])) | set(about_data.get("contact_info", []))
            )

            st.success("Done!")

            # ── Main page results ──
            st.header("Main Page")
            st.subheader("Page Title")
            st.write(main["title"])

            st.subheader("Meta Description")
            st.write(main["meta_description"])

            # ── About page results ──
            if about_data:
                st.header(f"About Page (`{about_url}`)")
                st.subheader("Page Title")
                st.write(about_data.get("title", "Not found"))
                st.subheader("Meta Description")
                st.write(about_data.get("meta_description", "Not found"))
            else:
                st.info("No /about or /about-us page found (or it returned an error).")

            # ── Key people (combined) ──
            st.header("Key People")
            if all_people:
                for person in all_people:
                    st.write(f"- {person}")
            else:
                st.write("No key people found on these pages.")

            # ── Contact info (combined) ──
            st.header("Contact / Address Info")
            if all_contact:
                for item in all_contact:
                    st.write(f"- {item}")
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
