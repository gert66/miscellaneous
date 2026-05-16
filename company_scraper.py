# Install dependencies:
#   pip install streamlit selenium requests
#   Chrome or Chromium must be installed on the system.
#   Selenium 4.6+ includes selenium-manager which downloads chromedriver automatically.
#   If auto-download fails, install manually: pip install webdriver-manager
#   and replace make_driver() with the webdriver-manager variant shown in the comment there.
#
# Run the app:
#   streamlit run company_scraper.py

import re
import time
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ── Constants ────────────────────────────────────────────────────────────────

PERSON_ROLE_RE = re.compile(
    r"\b(founder|co-founder|ceo|chief executive|president|cto|coo|cfo|"
    r"director|chairman|chairwoman|managing partner|partner|head of)\b",
    re.I,
)
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")

# JS that walks the live DOM and returns the most-specific element whose
# textContent includes the given string, plus its tag, attributes, and context.
_JS_FIND_EVIDENCE = """
(function(searchText) {
    var skip = {SCRIPT:1, STYLE:1, NOSCRIPT:1, HEAD:1, META:1, LINK:1};
    var root = document.body || document.documentElement;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, {
        acceptNode: function(n) {
            return skip[n.tagName] ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
        }
    });
    var best = null, bestLen = Infinity, node;
    while ((node = walker.nextNode())) {
        var txt = node.textContent || '';
        if (txt.indexOf(searchText) !== -1 && txt.length < bestLen && txt.length > 0) {
            best = node; bestLen = txt.length;
        }
    }
    if (!best) return null;
    var attrs = {};
    for (var i = 0; i < (best.attributes || []).length; i++) {
        var a = best.attributes[i];
        if (a.name === 'class' || a.name === 'id' || a.name === 'name' || a.name === 'itemtype') {
            attrs[a.name] = a.value;
        }
    }
    var ctx = best.textContent.trim();
    return {
        tag: best.tagName.toLowerCase(),
        attrs: attrs,
        context: ctx.length > 300 ? ctx.substring(0, 300) + '...' : ctx
    };
})(arguments[0])
"""

# ── Driver ───────────────────────────────────────────────────────────────────

def make_driver() -> webdriver.Chrome:
    # webdriver-manager alternative:
    #   from webdriver_manager.chrome import ChromeDriverManager
    #   from selenium.webdriver.chrome.service import Service
    #   return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


def load_page(driver: webdriver.Chrome, url: str, timeout: int = 20) -> None:
    driver.get(url)
    # Wait for the DOM to reach readyState=complete
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    # Brief pause to let any post-load JS settle (lazy-loaded content, etc.)
    time.sleep(1.5)


# ── Element / evidence helpers ───────────────────────────────────────────────

def _el_tag(el) -> str:
    """Build a '<tag attr="val">' string from a Selenium WebElement."""
    parts = []
    for attr in ("class", "id", "name", "itemtype"):
        val = el.get_attribute(attr)
        if val:
            parts.append(f'{attr}="{val}"')
    suffix = (" " + " ".join(parts)) if parts else ""
    return f"<{el.tag_name}{suffix}>"


def _el_evidence(el, source_url: str) -> dict:
    ctx = (el.text or "").strip()
    if len(ctx) > 300:
        ctx = ctx[:300] + "..."
    return {"tag": _el_tag(el), "context": ctx, "url": source_url}


def _js_evidence(driver: webdriver.Chrome, search_text: str, source_url: str) -> dict:
    """Use JS DOM traversal to locate the tightest element containing search_text."""
    result = driver.execute_script(_JS_FIND_EVIDENCE, search_text)
    if not result:
        return {"tag": "unknown", "context": "", "url": source_url}
    attr_str = " ".join(f'{k}="{v}"' for k, v in result["attrs"].items() if v)
    tag = f'<{result["tag"]}' + (f" {attr_str}" if attr_str else "") + ">"
    return {"tag": tag, "context": result["context"], "url": source_url}


def item(value: str, evidence: dict) -> dict:
    return {"value": value, "evidence": evidence}


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_contact(driver: webdriver.Chrome, source_url: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    def add(value: str, evidence: dict) -> None:
        if value not in seen:
            seen.add(value)
            results.append(item(value, evidence))

    page_source = driver.page_source

    # Emails — search page source so we catch mailto: and hidden attributes too
    for email in re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", page_source):
        add(f"Email: {email}", _js_evidence(driver, email, source_url))

    # Phone numbers — search visible body text to avoid false positives from scripts
    body_text = driver.find_element(By.TAG_NAME, "body").text
    for phone in re.findall(r"(?:\+?\d[\d\s\-().]{7,}\d)", body_text)[:5]:
        cleaned = phone.strip()
        if len(re.sub(r"\D", "", cleaned)) >= 7:
            add(f"Phone: {cleaned}", _js_evidence(driver, cleaned[:12], source_url))

    # <address> tags
    for el in driver.find_elements(By.TAG_NAME, "address"):
        text = el.text.strip()
        if text:
            add(f"Address: {text}", _el_evidence(el, source_url))

    # Elements whose class or id contains address / location / contact
    for sel in (
        "[class*='address']", "[class*='location']", "[class*='contact']",
        "[id*='address']",    "[id*='location']",    "[id*='contact']",
    ):
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            text = el.text.strip()
            if text and len(text) < 300:
                add(f"Contact block: {text}", _el_evidence(el, source_url))

    return results


def extract_people(driver: webdriver.Chrome, source_url: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    def add_person(name: str, role: str, evidence: dict) -> None:
        key = name.lower().strip()
        if not name or key in seen:
            return
        seen.add(key)
        label = name + (f" — {role}" if role else "")
        results.append(item(label, evidence))

    # 1. schema.org Person items (structured markup)
    schema_people = driver.execute_script("""
        var people = [];
        document.querySelectorAll('[itemtype*="schema.org/Person"]').forEach(function(el) {
            var nameEl = el.querySelector('[itemprop="name"]');
            var roleEl = el.querySelector('[itemprop="jobTitle"],[itemprop="role"]');
            if (nameEl) {
                var attrs = {};
                ['class','id'].forEach(function(a) { if (el.getAttribute(a)) attrs[a] = el.getAttribute(a); });
                var ctx = el.textContent.trim();
                people.push({
                    name: nameEl.textContent.trim(),
                    role: roleEl ? roleEl.textContent.trim() : '',
                    tag: el.tagName.toLowerCase(),
                    attrs: attrs,
                    context: ctx.length > 300 ? ctx.substring(0,300)+'...' : ctx
                });
            }
        });
        return people;
    """) or []
    for p in schema_people:
        attr_str = " ".join(f'{k}="{v}"' for k, v in p["attrs"].items() if v)
        tag = f'<{p["tag"]}' + (f" {attr_str}" if attr_str else "") + ">"
        ev = {"tag": tag, "context": p["context"], "url": source_url}
        add_person(p["name"], p["role"], ev)

    # 2. Team / bio card patterns — look for container elements with telling class names
    card_selectors = [
        "[class*='team-member']", "[class*='team_member']",
        "[class*='founder']",     "[class*='leadership']",
        "[class*='staff']",       "[class*='bio']",
        "[class*='person']",      "[class*='people']",
        "[class*='team-card']",   "[class*='member-card']",
    ]
    for sel in card_selectors:
        for card in driver.find_elements(By.CSS_SELECTOR, sel):
            heading = None
            for h_tag in ("h1", "h2", "h3", "h4", "h5", "strong", "b"):
                try:
                    heading = card.find_element(By.TAG_NAME, h_tag)
                    break
                except NoSuchElementException:
                    pass
            if not heading:
                continue
            name = heading.text.strip()
            if not NAME_RE.match(name) or len(name.split()) > 5:
                continue
            role = ""
            for role_sel in (
                "[class*='role']", "[class*='title']",
                "[class*='position']", "[class*='job']",
            ):
                try:
                    role = card.find_element(By.CSS_SELECTOR, role_sel).text.strip()
                    break
                except NoSuchElementException:
                    pass
            add_person(name, role, _el_evidence(card, source_url))

    # 3. Keyword fallback — scan visible lines for role words near capitalised names
    if not results:
        body_text = driver.find_element(By.TAG_NAME, "body").text
        for line in body_text.splitlines():
            if not PERSON_ROLE_RE.search(line):
                continue
            role_m = PERSON_ROLE_RE.search(line)
            role = role_m.group(0).title() if role_m else ""
            for name in NAME_RE.findall(line):
                if len(name.split()) >= 2:
                    add_person(name, role, _js_evidence(driver, name, source_url))

    return results


# ── Page-level scraping ──────────────────────────────────────────────────────

def scrape_loaded_page(driver: webdriver.Chrome, url: str) -> dict:
    """Extract all data from the page currently loaded in driver."""
    title = (driver.title or "Not found").strip()

    meta_desc = ""
    meta_evidence = None
    for sel in ('meta[name="description"]', 'meta[property="og:description"]'):
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            content = el.get_attribute("content") or ""
            if content.strip():
                meta_desc = content.strip()
                meta_evidence = {
                    "tag": _el_tag(el),
                    "context": f'content="{meta_desc}"',
                    "url": url,
                }
                break
        except NoSuchElementException:
            pass

    return {
        "title": title,
        "meta_description": meta_desc or "Not found",
        "meta_evidence": meta_evidence,
        "contact_info": extract_contact(driver, url),
        "people": extract_people(driver, url),
        "url": url,
    }


def find_about_url(driver: webdriver.Chrome, base_url: str) -> str | None:
    """Find an /about or /about-us URL from links on the currently loaded page."""
    about_re = re.compile(r"/about(?:-us)?/?$", re.I)

    # Check anchor tags on the page
    for el in driver.find_elements(By.TAG_NAME, "a"):
        href = el.get_attribute("href") or ""
        if href and about_re.search(urlparse(href).path):
            return href

    # Fall back to probing common paths directly
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    ua = "Mozilla/5.0 (compatible; CompanyScraper/1.0)"
    for candidate in ("/about", "/about-us"):
        try:
            r = requests.head(root + candidate, headers={"User-Agent": ua},
                              timeout=6, allow_redirects=True)
            if r.status_code < 400:
                return root + candidate
        except Exception:
            pass
    return None


def merge_items(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for it in a + b:
        k = it["value"].lower().strip()
        if k not in seen:
            seen.add(k)
            merged.append(it)
    return merged


# ── UI helpers ───────────────────────────────────────────────────────────────

def render_evidence_expander(evidence: dict) -> None:
    with st.expander("Source Evidence"):
        st.markdown(f"**Source URL:** {evidence['url']}")
        st.markdown(f"**HTML element:** `{evidence['tag']}`")
        if evidence.get("context"):
            st.markdown("**Surrounding text:**")
            st.code(evidence["context"], language=None)


def render_items(items: list[dict]) -> None:
    for it in items:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"- {it['value']}")
        with col2:
            render_evidence_expander(it["evidence"])


# ── App ──────────────────────────────────────────────────────────────────────

st.title("Company Info Scraper")
st.write(
    "Enter a company website URL. A headless Chrome browser will load the page "
    "(including JavaScript-rendered content) before extracting info."
)

url = st.text_input("Company Website URL", placeholder="https://example.com")

if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        driver: webdriver.Chrome | None = None
        try:
            with st.spinner("Starting headless Chrome..."):
                driver = make_driver()

            with st.spinner("Loading main page and waiting for JavaScript..."):
                load_page(driver, url)

            with st.spinner("Scraping main page..."):
                about_url = find_about_url(driver, url)
                main = scrape_loaded_page(driver, url)

            about_data: dict = {}
            if about_url and about_url.rstrip("/") != url.rstrip("/"):
                with st.spinner(f"Loading About page ({about_url})..."):
                    load_page(driver, about_url)
                with st.spinner("Scraping About page..."):
                    about_data = scrape_loaded_page(driver, about_url)

            all_people  = merge_items(main.get("people", []),       about_data.get("people", []))
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
                st.header("About Page")
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

        except WebDriverException as e:
            st.error(
                f"Chrome could not start: {e}\n\n"
                "Make sure Chrome or Chromium is installed and accessible in PATH."
            )
        except TimeoutException:
            st.error("Timed out waiting for the page to load.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            if driver:
                driver.quit()
