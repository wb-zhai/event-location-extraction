#!/usr/bin/env python3

from pathlib import Path
import argparse
from playwright.sync_api import sync_playwright


PRINT_CSS = r"""
@page {
    size: 16in 9in;
    margin: 0;
}

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    width: 16in !important;
    height: auto !important;
    overflow: visible !important;
    background: white !important;
}

.deck {
    width: 16in !important;
    height: auto !important;
    position: relative !important;
}

.slide {
    position: relative !important;
    inset: auto !important;

    display: flex !important;
    flex-direction: column !important;

    width: 16in !important;
    height: 9in !important;
    min-height: 9in !important;
    max-height: 9in !important;

    padding: 42px 54px 34px !important;

    page-break-after: always !important;
    break-after: page !important;

    overflow: hidden !important;
    box-shadow: none !important;
}

.slide:last-of-type {
    page-break-after: auto !important;
    break-after: auto !important;
}

.progress {
    display: none !important;
}

.nav-help {
    display: none !important;
}

.image-card,
.text-card,
.closing-card {
    box-shadow: none !important;
}

* {
    animation: none !important;
    transition: none !important;
}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Global V4 HTML presentation to a 16:9 PDF."
    )

    parser.add_argument(
        "html",
        type=Path,
        nargs="?",
        default=Path("global_v4_extraction_presentation.html"),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("global_v4_extraction_presentation.pdf"),
    )

    args = parser.parse_args()

    html_path = args.html.resolve()
    output_path = args.output.resolve()

    if not html_path.is_file():
        raise SystemExit(
            f"HTML file not found: {html_path}"
        )

    print(f"Input : {html_path}")
    print(f"Output: {output_path}")

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1600,
                "height": 900,
            }
        )

        # Open the local HTML.
        page.goto(
            html_path.as_uri(),
            wait_until="networkidle",
        )

        # Wait until all local PNGs are fully loaded.
        page.wait_for_function(
            """
            () => Array.from(document.images).every(
                img => img.complete && img.naturalWidth > 0
            )
            """
        )

        # Use print media rules.
        page.emulate_media(
            media="print"
        )

        # Force one exact 16:9 page per slide.
        page.add_style_tag(
            content=PRINT_CSS
        )

        # Give Chromium a moment to finish layout.
        page.wait_for_timeout(500)

        slide_count = page.locator(
            ".slide"
        ).count()

        print(
            f"Slides detected: {slide_count}"
        )

        page.pdf(
            path=str(output_path),

            width="16in",
            height="9in",

            margin={
                "top": "0",
                "right": "0",
                "bottom": "0",
                "left": "0",
            },

            print_background=True,

            prefer_css_page_size=True,

            display_header_footer=False,
        )

        browser.close()

    print()
    print("PDF created successfully:")
    print(output_path)


if __name__ == "__main__":
    main()
