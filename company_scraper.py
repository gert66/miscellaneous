# Install dependencies:
#   pip install streamlit requests beautifulsoup4
#
# Run the app:
#   streamlit run company_scraper.py

import re
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.title("Company Info Scraper")
st.write("Paste a company website URL below to extract basic information.")

url = st.text_input("Company Website URL", placeholder="https://example.com")

def fetch_company_info(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CompanyScraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else "Not found"

    meta_desc = ""
    for attrs in [{"name": "description"}, {"property": "og:description"}]:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            meta_desc = tag["content"].strip()
            break
    if not meta_desc:
        meta_desc = "Not found"

    contact_info = set()

    # Emails
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", response.text)
    for email in emails:
        contact_info.add(f"Email: {email}")

    # Phone numbers (common formats)
    phones = re.findall(r"(?:\+?\d[\d\s\-().]{7,}\d)", response.text)
    for phone in phones[:5]:  # limit noise
        cleaned = phone.strip()
        if len(re.sub(r"\D", "", cleaned)) >= 7:
            contact_info.add(f"Phone: {cleaned}")

    # Address hints: look for <address> tags or common address containers
    address_tags = soup.find_all("address")
    for tag in address_tags:
        text = tag.get_text(separator=" ", strip=True)
        if text:
            contact_info.add(f"Address: {text}")

    # Also check divs/spans with address-related class/id names
    address_keywords = re.compile(r"address|location|contact", re.I)
    for tag in soup.find_all(["div", "span", "p", "section"], class_=address_keywords):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) < 300:
            contact_info.add(f"Contact block: {text}")

    return {
        "title": title,
        "meta_description": meta_desc,
        "contact_info": sorted(contact_info) if contact_info else ["No contact info found"],
    }


if st.button("Get Company Info"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        with st.spinner("Fetching page..."):
            try:
                info = fetch_company_info(url)
                st.success("Done!")

                st.subheader("Page Title")
                st.write(info["title"])

                st.subheader("Meta Description")
                st.write(info["meta_description"])

                st.subheader("Contact / Address Info")
                for item in info["contact_info"]:
                    st.write(f"- {item}")

            except requests.exceptions.MissingSchema:
                st.error("Invalid URL. Make sure it starts with http:// or https://")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the URL. Check the address and try again.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error: {e}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
