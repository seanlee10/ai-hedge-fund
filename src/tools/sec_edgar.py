"""SEC EDGAR API client for fetching insider trading data from Form 4 filings."""

import os
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional
from functools import lru_cache

from src.data.cache import get_cache
from src.data.models import InsiderTrade

# Global cache instance
_cache = get_cache()

# SEC EDGAR rate limit: 10 requests per second
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests

# User agent required by SEC
_USER_AGENT = os.environ.get("SEC_USER_AGENT", "AI-Hedge-Fund contact@example.com")


def _rate_limit():
    """Ensure we don't exceed SEC's rate limit."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _make_sec_request(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Make SEC EDGAR request with rate limiting and retries."""
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }

    for attempt in range(max_retries + 1):
        _rate_limit()

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return response
            elif response.status_code == 429 and attempt < max_retries:
                delay = 5 * (attempt + 1)
                print(f"SEC rate limited. Waiting {delay}s...")
                time.sleep(delay)
                continue
            elif response.status_code == 404:
                return None
            else:
                print(f"SEC request failed: {response.status_code}")
                return None
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"SEC request error: {e}")
            return None

    return None


@lru_cache(maxsize=1000)
def _get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Get CIK (Central Index Key) for a ticker symbol."""
    # First try the company tickers JSON
    url = "https://www.sec.gov/files/company_tickers.json"
    response = _make_sec_request(url)

    if response:
        try:
            data = response.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry.get("cik_str", ""))
                    return cik.zfill(10)  # Pad to 10 digits
        except Exception:
            pass

    return None


def _parse_form4_xml(xml_content: str, ticker: str) -> list[InsiderTrade]:
    """Parse Form 4 XML and extract insider trade information."""
    trades = []

    try:
        root = ET.fromstring(xml_content)

        # Get issuer info
        issuer = root.find(".//issuer")
        issuer_name = ""
        if issuer is not None:
            issuer_name_elem = issuer.find("issuerName")
            if issuer_name_elem is not None:
                issuer_name = issuer_name_elem.text or ""

        # Get reporting owner info
        owner = root.find(".//reportingOwner")
        owner_name = ""
        owner_title = ""
        is_director = False

        if owner is not None:
            owner_id = owner.find("reportingOwnerId")
            if owner_id is not None:
                name_elem = owner_id.find("rptOwnerName")
                if name_elem is not None:
                    owner_name = name_elem.text or ""

            relationship = owner.find("reportingOwnerRelationship")
            if relationship is not None:
                is_director_elem = relationship.find("isDirector")
                if is_director_elem is not None:
                    is_director = is_director_elem.text == "1" or is_director_elem.text == "true"

                title_elem = relationship.find("officerTitle")
                if title_elem is not None:
                    owner_title = title_elem.text or ""

        # Get period of report (filing date)
        period_elem = root.find(".//periodOfReport")
        filing_date = ""
        if period_elem is not None:
            filing_date = period_elem.text or ""

        # Parse non-derivative transactions
        for trans in root.findall(".//nonDerivativeTransaction"):
            trade = _parse_transaction(
                trans, ticker, issuer_name, owner_name, owner_title, is_director, filing_date
            )
            if trade:
                trades.append(trade)

        # Parse derivative transactions
        for trans in root.findall(".//derivativeTransaction"):
            trade = _parse_transaction(
                trans, ticker, issuer_name, owner_name, owner_title, is_director, filing_date
            )
            if trade:
                trades.append(trade)

    except ET.ParseError as e:
        print(f"XML parse error: {e}")
    except Exception as e:
        print(f"Error parsing Form 4: {e}")

    return trades


def _parse_transaction(
    trans: ET.Element,
    ticker: str,
    issuer: str,
    owner_name: str,
    owner_title: str,
    is_director: bool,
    filing_date: str,
) -> Optional[InsiderTrade]:
    """Parse a single transaction element."""
    try:
        # Get security title
        security_elem = trans.find(".//securityTitle/value")
        security_title = security_elem.text if security_elem is not None else None

        # Get transaction date
        trans_date_elem = trans.find(".//transactionDate/value")
        trans_date = trans_date_elem.text if trans_date_elem is not None else None

        # Get transaction amounts
        amounts = trans.find(".//transactionAmounts")
        if amounts is None:
            return None

        # Get shares
        shares_elem = amounts.find(".//transactionShares/value")
        shares = None
        if shares_elem is not None and shares_elem.text:
            try:
                shares = float(shares_elem.text)
            except ValueError:
                pass

        # Get price per share
        price_elem = amounts.find(".//transactionPricePerShare/value")
        price = None
        if price_elem is not None and price_elem.text:
            try:
                price = float(price_elem.text)
            except ValueError:
                pass

        # Get acquisition/disposition code (A = acquired, D = disposed)
        code_elem = amounts.find(".//transactionAcquiredDisposedCode/value")
        if code_elem is not None:
            code = code_elem.text
            if code == "D" and shares is not None:
                shares = -abs(shares)  # Negative for sales
            elif code == "A" and shares is not None:
                shares = abs(shares)  # Positive for purchases

        # Get post-transaction holdings
        post_holding = trans.find(".//postTransactionAmounts")
        shares_after = None
        if post_holding is not None:
            shares_after_elem = post_holding.find(".//sharesOwnedFollowingTransaction/value")
            if shares_after_elem is not None and shares_after_elem.text:
                try:
                    shares_after = float(shares_after_elem.text)
                except ValueError:
                    pass

        # Calculate shares before and transaction value
        shares_before = None
        if shares_after is not None and shares is not None:
            shares_before = shares_after - shares

        transaction_value = None
        if shares is not None and price is not None:
            transaction_value = abs(shares) * price

        return InsiderTrade(
            ticker=ticker,
            issuer=issuer,
            name=owner_name,
            title=owner_title,
            is_board_director=is_director,
            transaction_date=trans_date,
            transaction_shares=shares,
            transaction_price_per_share=price,
            transaction_value=transaction_value,
            shares_owned_before_transaction=shares_before,
            shares_owned_after_transaction=shares_after,
            security_title=security_title,
            filing_date=filing_date,
        )
    except Exception as e:
        print(f"Error parsing transaction: {e}")
        return None


def get_insider_trades_sec(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from SEC EDGAR Form 4 filings."""
    cache_key = f"sec_{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # Get CIK for the ticker
    cik = _get_cik_for_ticker(ticker)
    if not cik:
        print(f"Could not find CIK for ticker: {ticker}")
        return []

    # Fetch company submissions to get recent filings
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = _make_sec_request(submissions_url)

    if not response:
        return []

    try:
        data = response.json()
    except Exception:
        return []

    # Get recent filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    # Parse date filters
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None

    all_trades = []
    form4_count = 0

    for i, form in enumerate(forms):
        if form != "4":  # Only Form 4 (insider trading)
            continue

        filing_date = filing_dates[i] if i < len(filing_dates) else ""
        if not filing_date:
            continue

        try:
            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
        except ValueError:
            continue

        # Filter by date range
        if filing_dt > end_dt:
            continue
        if start_dt and filing_dt < start_dt:
            continue

        # Get accession number and primary document
        accession = accession_numbers[i] if i < len(accession_numbers) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""

        if not accession or not primary_doc:
            continue

        # Format accession number for URL (remove dashes)
        accession_formatted = accession.replace("-", "")

        # Strip XSL transformation prefix if present (e.g., "xslF345X05/")
        # The raw XML is available without this prefix
        if "/" in primary_doc:
            primary_doc = primary_doc.split("/")[-1]

        # Fetch the Form 4 XML
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_formatted}/{primary_doc}"
        filing_response = _make_sec_request(filing_url)

        if filing_response:
            trades = _parse_form4_xml(filing_response.text, ticker)
            all_trades.extend(trades)
            form4_count += 1

        # Limit the number of Form 4s we fetch
        if form4_count >= 50 or len(all_trades) >= limit:
            break

    # Sort by filing date descending
    all_trades.sort(key=lambda t: t.filing_date or "", reverse=True)

    # Limit results
    all_trades = all_trades[:limit]

    if all_trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in all_trades])

    return all_trades
