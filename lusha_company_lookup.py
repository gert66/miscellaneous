import streamlit as st
import requests

# ── Configuratie ──────────────────────────────────────────────────────────────
LUSHA_API_KEY = "JOUW_API_KEY_HIER"   # <-- vervang dit door je Lusha API-key
LUSHA_ENDPOINT = "https://api.lusha.com/company"
# ─────────────────────────────────────────────────────────────────────────────


def lookup_company(domain: str, api_key: str) -> dict:
    headers = {
        "api_key": api_key,
        "Content-Type": "application/json",
    }
    params = {"domain": domain}
    response = requests.get(LUSHA_ENDPOINT, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def render_field(label: str, value) -> None:
    if value is not None and value != "":
        st.markdown(f"**{label}:** {value}")


def render_company_info(data: dict) -> None:
    company = data.get("company", data)  # sommige Lusha-responses zitten in 'company'

    st.subheader("Bedrijfsinformatie")

    col1, col2 = st.columns(2)
    with col1:
        render_field("Naam", company.get("name"))
        render_field("Domein", company.get("domain"))
        render_field("Website", company.get("website"))
        render_field("Industrie", company.get("industry"))
        render_field("Sub-industrie", company.get("subIndustry"))
        render_field("Bedrijfstype", company.get("companyType"))
    with col2:
        render_field("Aantal medewerkers", company.get("employeeCount"))
        render_field("Medewerkersbereik", company.get("employeeCountRange"))
        render_field("Omzet", company.get("revenue"))
        render_field("Opgericht", company.get("foundedYear"))
        render_field("Linkedin", company.get("linkedinUrl"))
        render_field("Telefoonnummer", company.get("phoneNumber"))

    # Locatie
    location = company.get("headquarters") or company.get("location")
    if location:
        st.subheader("Locatie")
        if isinstance(location, dict):
            render_field("Adres", location.get("address"))
            render_field("Stad", location.get("city"))
            render_field("Staat/Provincie", location.get("state"))
            render_field("Land", location.get("country"))
            render_field("Postcode", location.get("zipCode"))
        else:
            st.markdown(f"**Locatie:** {location}")

    # Beschrijving
    description = company.get("description")
    if description:
        st.subheader("Beschrijving")
        st.write(description)

    # Ruwe JSON voor transparantie
    with st.expander("Volledige JSON-response"):
        st.json(data)


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Lusha Bedrijfsopzoeking", page_icon="🔍", layout="centered")
st.title("🔍 Lusha Bedrijfsopzoeking")
st.caption("Voer een bedrijfsdomein in om bedrijfsinformatie op te halen via de Lusha API.")

# Optioneel: API-key overschrijven via de sidebar (handig bij demos)
with st.sidebar:
    st.header("Instellingen")
    api_key_input = st.text_input(
        "Lusha API-key",
        value=LUSHA_API_KEY,
        type="password",
        help="Je Lusha API-key. Standaard wordt de waarde uit de code gebruikt.",
    )

domain_input = st.text_input(
    "Bedrijfsdomein",
    placeholder="bijv. apple.com",
    help="Voer het domein in zonder 'https://' of 'www.'",
)

if st.button("Opzoeken", type="primary", use_container_width=True):
    active_key = api_key_input.strip()
    domain = domain_input.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")

    if not active_key or active_key == "JOUW_API_KEY_HIER":
        st.error("Vul eerst een geldige Lusha API-key in (sidebar of bovenaan de code).")
    elif not domain:
        st.warning("Vul een bedrijfsdomein in.")
    else:
        with st.spinner(f"Bedrijfsinformatie ophalen voor **{domain}**…"):
            try:
                result = lookup_company(domain, active_key)
                render_company_info(result)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if status == 401:
                    st.error("API-key ongeldig of verlopen (401 Unauthorized).")
                elif status == 404:
                    st.warning(f"Geen bedrijf gevonden voor domein **{domain}** (404).")
                elif status == 429:
                    st.error("Te veel verzoeken. Wacht even en probeer opnieuw (429 Too Many Requests).")
                else:
                    st.error(f"HTTP-fout {status}: {e}")
            except requests.ConnectionError:
                st.error("Kan geen verbinding maken met de Lusha API. Controleer je internetverbinding.")
            except requests.Timeout:
                st.error("Het verzoek aan de Lusha API duurde te lang (timeout).")
            except Exception as e:
                st.error(f"Onverwachte fout: {e}")
