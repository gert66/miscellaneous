"""
Human-behaviour web scraper using Playwright.
Mimics real user interactions to bypass bot detection.
"""

import os
import random
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

_VIEWPORTS = [
    {"width": 1280, "height": 800},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
]

_COOKIE_CONSENT_TEXTS = [
    "Accept",
    "Accepteer",
    "Akkoord",
    "Allow all",
    "Accept all",
    "Tout accepter",
]

_STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    window.chrome = {runtime: {}};
"""


def _is_server_mode() -> bool:
    return bool(os.environ.get("STREAMLIT_SERVER_PORT"))


def _simulate_mouse_path(page, target_x: int, target_y: int) -> None:
    start_x = random.randint(0, target_x // 2)
    start_y = random.randint(0, target_y // 2)
    steps = random.randint(5, 8)

    for i in range(1, steps + 1):
        t = i / steps
        jitter_x = random.randint(-15, 15)
        jitter_y = random.randint(-15, 15)
        x = int(start_x + (target_x - start_x) * t + jitter_x)
        y = int(start_y + (target_y - start_y) * t + jitter_y)
        page.mouse.move(x, y)
        time.sleep(random.uniform(0.05, 0.10))


def _handle_cookie_consent(page) -> None:
    for text in _COOKIE_CONSENT_TEXTS:
        try:
            btn = page.locator(f"button:has-text('{text}'), a:has-text('{text}')").first
            if btn.is_visible(timeout=1000):
                btn.click()
                time.sleep(1.0)
                return
        except Exception:
            continue


def _simulate_human_behaviour(page, viewport: dict) -> None:
    # Wait after page load
    time.sleep(random.uniform(1.5, 3.5))

    # Handle cookie consent before interacting
    _handle_cookie_consent(page)

    # Move mouse in a curved path to page center
    center_x = viewport["width"] // 2
    center_y = viewport["height"] // 2
    _simulate_mouse_path(page, center_x, center_y)

    # Scroll down slowly
    scroll_steps = random.randint(3, 5)
    for _ in range(scroll_steps):
        delta = random.randint(200, 400)
        page.mouse.wheel(0, delta)
        time.sleep(random.uniform(0.4, 0.8))

    # Pause
    time.sleep(random.uniform(1.0, 2.0))

    # Scroll back up slightly
    page.mouse.wheel(0, -random.randint(100, 200))

    # Final wait
    time.sleep(random.uniform(0.5, 1.5))


def scrape_with_human_behaviour(url: str, max_chars: int = 6000) -> dict:
    """
    Scrape a URL using Playwright with realistic human behaviour to bypass bot detection.

    Returns a dict with keys: text, title, final_url, status_code, success, method, error.
    """
    viewport = random.choice(_VIEWPORTS)
    user_agent = random.choice(_USER_AGENTS)
    headless = _is_server_mode()

    with sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=user_agent,
                viewport=viewport,
                locale="en-US",
                timezone_id="Europe/Amsterdam",
                geolocation={"latitude": 52.37, "longitude": 4.89},
                permissions=["geolocation"],
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            page = context.new_page()
            page.add_init_script(_STEALTH_SCRIPT)

            response = page.goto(url, timeout=30_000, wait_until="domcontentloaded")
            status_code = response.status if response else 0

            _simulate_human_behaviour(page, viewport)

            body_text = page.inner_text("body")
            # Normalise whitespace
            body_text = " ".join(body_text.split())
            title = page.title()
            final_url = page.url

            return {
                "text": body_text[:max_chars],
                "title": title,
                "final_url": final_url,
                "status_code": status_code,
                "success": True,
                "method": "human_playwright",
                "error": "",
            }

        except PlaywrightTimeoutError:
            return {
                "text": "",
                "title": "",
                "final_url": url,
                "status_code": 0,
                "success": False,
                "method": "human_playwright",
                "error": "timeout",
            }
        except Exception as e:
            return {
                "text": "",
                "title": "",
                "final_url": url,
                "status_code": 0,
                "success": False,
                "method": "human_playwright",
                "error": str(e),
            }
        finally:
            if browser:
                browser.close()


if __name__ == "__main__":
    result = scrape_with_human_behaviour("https://www.adidas.com")
    print(f"Success: {result['success']}")
    print(f"Title: {result['title']}")
    print(f"Text preview: {result['text'][:500]}")

    result2 = scrape_with_human_behaviour("https://www.abnamro.nl")
    print(f"ABN AMRO Success: {result2['success']}")
    print(f"ABN AMRO Title: {result2['title']}")
