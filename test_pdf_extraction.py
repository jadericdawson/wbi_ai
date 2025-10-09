#!/usr/bin/env python3
"""
Quick test script to verify AI-powered PDF extraction system.

Usage:
    python test_pdf_extraction.py <sam_gov_url>

Example:
    python test_pdf_extraction.py "https://sam.gov/opp/abc123/view"
"""

import sys
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_playwright_scrape(url: str):
    """Test the scrape_with_playwright function"""
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    logger.info(f"Testing Playwright scrape on: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate
            logger.info("  Navigating to page...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)

            # Get content
            html_content = page.content()
            page_text = page.inner_text("body")

            logger.info(f"  Page loaded: {len(page_text)} chars")

            # Test Layer 1: HTML selectors
            logger.info("  Testing Layer 1: HTML selectors...")

            pdf_links = page.query_selector_all('a[href*=".pdf"]')
            logger.info(f"    Found {len(pdf_links)} direct PDF links")

            attachment_rows = page.query_selector_all('tr[class*="attachment"], tbody tr')
            logger.info(f"    Found {len(attachment_rows)} potential attachment rows")

            download_buttons = page.query_selector_all('button[title*="Download"], a[aria-label*="Download"]')
            logger.info(f"    Found {len(download_buttons)} download buttons")

            # Check if we need Layer 2
            found_any = len(pdf_links) > 0 or len(attachment_rows) > 0 or len(download_buttons) > 0

            if not found_any and "Attachments" in page_text:
                logger.info("  Testing Layer 2: AI HTML analysis...")

                # Extract attachments section
                soup = BeautifulSoup(html_content, 'html.parser')
                attachments_section = soup.find(string=lambda text: text and 'Attachments' in text)

                if attachments_section:
                    container = attachments_section.find_parent(['div', 'section', 'table'])
                    if container:
                        attachments_html = str(container)[:10000]
                        logger.info(f"    Extracted attachments HTML: {len(attachments_html)} chars")

                        # Show sample
                        logger.info("    Sample HTML:")
                        print("\n" + "="*80)
                        print(attachments_html[:500])
                        print("="*80 + "\n")

                        logger.warning("    ⚠️  To test AI analysis, you need GPT-4.1 API credentials")
                        logger.info("    💡 The system would send this HTML to GPT-4.1 for URL extraction")
                    else:
                        logger.error("    ❌ Could not find attachments container")
                else:
                    logger.error("    ❌ Could not find 'Attachments' section in HTML")
            elif not found_any:
                logger.warning("  ⚠️  No PDFs found and no 'Attachments' section in text")
            else:
                logger.info(f"  ✅ Layer 1 successful: Found {len(pdf_links) + len(attachment_rows) + len(download_buttons)} potential PDFs")

            browser.close()

    except Exception as e:
        logger.error(f"  ❌ Test failed: {e}")
        return False

    return True


def test_ai_analysis_sample():
    """Show example of AI analysis prompt"""
    sample_html = """
    <div class="attachments-section">
        <h3>Attachments (2)</h3>
        <table>
            <tr>
                <td>
                    <span class="file-name">J_A Auto Notify Services_Redacted.pdf</span>
                    <span class="file-size">240 KB</span>
                </td>
                <td>
                    <button class="btn-download" data-file-url="/api/file/download/67890">
                        Download
                    </button>
                </td>
            </tr>
            <tr>
                <td>
                    <span class="file-name">Pricing_Schedule.pdf</span>
                    <span class="file-size">85 KB</span>
                </td>
                <td>
                    <button onclick="downloadFile('/download/12345')">
                        Download
                    </button>
                </td>
            </tr>
        </table>
    </div>
    """

    ai_prompt = f"""You are an expert at analyzing HTML to find file download URLs.

HTML SNIPPET (Attachments section from SAM.gov):
{sample_html}

TASK:
1. Find all PDF file names mentioned
2. For each PDF, find download URL by looking for:
   - <a href="..."> tags
   - onclick handlers with URLs
   - data-* attributes with file paths
   - API endpoints or download URLs

3. SAM.gov common patterns:
   - URLs might be relative (start with /) or absolute
   - Downloads may use /api/file/download/ endpoints
   - Look for patterns like: /download/{{file_id}}

Respond with JSON:
{{
  "pdfs_found": [
    {{
      "filename": "exact filename from HTML",
      "download_url": "full or relative URL",
      "extraction_method": "how you found it"
    }}
  ],
  "notes": "any observations"
}}
"""

    logger.info("Example AI Analysis Prompt:")
    print("\n" + "="*80)
    print(ai_prompt)
    print("="*80 + "\n")

    logger.info("Expected AI Response:")
    expected_response = {
        "pdfs_found": [
            {
                "filename": "J_A Auto Notify Services_Redacted.pdf",
                "download_url": "/api/file/download/67890",
                "extraction_method": "data-file-url attribute on download button"
            },
            {
                "filename": "Pricing_Schedule.pdf",
                "download_url": "/download/12345",
                "extraction_method": "onclick handler - downloadFile('/download/12345')"
            }
        ],
        "notes": "Found 2 PDFs with download URLs in button attributes. URLs are relative and will need to be made absolute."
    }

    print("\n" + "="*80)
    print(json.dumps(expected_response, indent=2))
    print("="*80 + "\n")


def main():
    logger.info("="*80)
    logger.info("PDF Extraction System Test")
    logger.info("="*80)

    if len(sys.argv) > 1:
        url = sys.argv[1]
        logger.info(f"\nTesting with URL: {url}\n")
        test_playwright_scrape(url)
    else:
        logger.info("\nNo URL provided - showing example AI analysis\n")
        test_ai_analysis_sample()
        logger.info("\nTo test with real URL, run:")
        logger.info('  python test_pdf_extraction.py "https://sam.gov/opp/abc123/view"')


if __name__ == "__main__":
    main()
