# -*- coding: utf-8 -*-
"""Делает PNG-скриншоты дашборда Streamlit с помощью Playwright."""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 2400},
            device_scale_factor=2,
        )
        page = await ctx.new_page()
        await page.goto("http://localhost:8501", wait_until="networkidle",
                        timeout=60_000)
        # Streamlit отдаёт страницу инкрементально — даём время на отрисовку.
        await page.wait_for_timeout(9_000)
        # Прокручиваем, чтобы вызвать ленивую отрисовку Plotly во всех блоках.
        for offset in (0, 800, 1600, 2400, 3200, 0):
            await page.evaluate(f"window.scrollTo(0, {offset})")
            await page.wait_for_timeout(1_000)
        await page.set_viewport_size({"width": 1600, "height": 2400})
        await page.wait_for_timeout(2_000)
        # Полностраничный скриншот — это и есть дашборд целиком.
        await page.screenshot(path=str(out_dir / "dashboard_full.png"),
                              full_page=True)
        # Также сохраним «верх» и «низ» по виду экрана.
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1_500)
        await page.screenshot(path=str(out_dir / "dashboard_top.png"),
                              full_page=False, clip={"x": 0, "y": 0,
                                                     "width": 1600,
                                                     "height": 1100})
        await browser.close()
        print("Saved:", *(p.name for p in sorted(out_dir.glob("dashboard*.png"))))


if __name__ == "__main__":
    asyncio.run(main())
