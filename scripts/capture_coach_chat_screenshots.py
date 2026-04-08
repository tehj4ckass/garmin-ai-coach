#!/usr/bin/env python3
"""
Start Coach Chat (Chainlit) headless, capture 1-2 PNGs for docs/screenshots/.

Usage (from repo root):
  python scripts/capture_coach_chat_screenshots.py

Requires: playwright + chromium (`python -m playwright install chromium`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("SCREENSHOT_CHAINLIT_PORT", "18765"))
BASE = f"http://127.0.0.1:{PORT}"


def _wait_http_ok(url: str, timeout_sec: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise TimeoutError(f"Server did not respond in time: {url}")


def main() -> int:
    out_dir = REPO_ROOT / "docs" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        str(REPO_ROOT / "cli" / "qa_chainlit_app.py"),
        "--headless",
        "--port",
        str(PORT),
    ]
    env = os.environ.copy()
    env.setdefault("COACH_CONFIG", str(REPO_ROOT / "coach_config.yaml"))

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_http_ok(BASE + "/", timeout_sec=90.0)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.goto(BASE + "/", wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(out_dir / "coach_chat_welcome.png"))

            # If the welcome table lists at least one run, select "1" to show the confirmation UI (no LLM call).
            try:
                if page.locator("textarea").count() > 0:
                    composer = page.locator("textarea").first
                    composer.click()
                    composer.fill("1")
                    composer.press("Enter")
                    page.wait_for_timeout(1500)
                    page.get_by_text("Kontext geladen").first.wait_for(timeout=15000)
                    page.wait_for_timeout(500)
                    page.screenshot(path=str(out_dir / "coach_chat_run_selected.png"))
            except Exception as e:
                print(f"Note: second screenshot skipped ({e!s})", file=sys.stderr)

            browser.close()

        print(f"Wrote: {out_dir / 'coach_chat_welcome.png'}")
        p2 = out_dir / "coach_chat_run_selected.png"
        if p2.exists():
            print(f"Wrote: {p2}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
