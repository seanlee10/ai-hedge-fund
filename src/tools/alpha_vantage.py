"""Alpha Vantage API client with rate limiting for Premium tier (75 calls/min)."""

import os
import time
import requests
from collections import deque
from datetime import datetime
from typing import Optional

from src.data.cache import get_cache
from src.data.models import (
    Price,
    FinancialMetrics,
    LineItem,
    CompanyNews,
)

# Global cache instance
_cache = get_cache()

# Rate limiting: track timestamps of recent API calls
_call_timestamps: deque = deque(maxlen=75)
_RATE_LIMIT = 75  # calls per minute for Premium tier
_RATE_WINDOW = 60  # seconds


def _check_rate_limit():
    """Sleep if approaching rate limit."""
    now = time.time()
    # Remove timestamps older than the rate window
    while _call_timestamps and now - _call_timestamps[0] > _RATE_WINDOW:
        _call_timestamps.popleft()

    # If we're at the limit, wait until the oldest call expires
    if len(_call_timestamps) >= _RATE_LIMIT - 5:  # Leave buffer of 5
        sleep_time = _RATE_WINDOW - (now - _call_timestamps[0]) + 0.1
        if sleep_time > 0:
            print(f"Rate limit approaching, sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)


def _make_av_request(params: dict, max_retries: int = 3) -> dict:
    """Make Alpha Vantage API request with rate limiting and retries."""
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

    params["apikey"] = api_key
    base_url = "https://www.alphavantage.co/query"

    for attempt in range(max_retries + 1):
        _check_rate_limit()
        _call_timestamps.append(time.time())

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            # Check for rate limit error in response
            if "Note" in data and "API call frequency" in data.get("Note", ""):
                if attempt < max_retries:
                    delay = 60 + (30 * attempt)
                    print(f"Rate limited. Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s...")
                    time.sleep(delay)
                    continue
            # Check for error messages
            if "Error Message" in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return {}
            return data

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s...")
            time.sleep(delay)
            continue

        print(f"Alpha Vantage request failed: {response.status_code}")
        return {}

    return {}


def get_prices_av(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily price data from Alpha Vantage."""
    cache_key = f"av_{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
    }

    data = _make_av_request(params)
    if not data or "Time Series (Daily)" not in data:
        return []

    time_series = data["Time Series (Daily)"]
    prices = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    for date_str, values in time_series.items():
        date_dt = datetime.strptime(date_str, "%Y-%m-%d")
        if start_dt <= date_dt <= end_dt:
            prices.append(Price(
                time=date_str,
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=int(float(values["6. volume"])),
            ))

    # Sort by date ascending
    prices.sort(key=lambda p: p.time)

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])

    return prices


def _get_overview(ticker: str) -> dict:
    """Fetch company overview data."""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
    }
    return _make_av_request(params)


def _get_income_statement(ticker: str) -> dict:
    """Fetch income statement data."""
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
    }
    return _make_av_request(params)


def _get_balance_sheet(ticker: str) -> dict:
    """Fetch balance sheet data."""
    params = {
        "function": "BALANCE_SHEET",
        "symbol": ticker,
    }
    return _make_av_request(params)


def _get_cash_flow(ticker: str) -> dict:
    """Fetch cash flow statement data."""
    params = {
        "function": "CASH_FLOW",
        "symbol": ticker,
    }
    return _make_av_request(params)


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Safely convert string to float, returning None for invalid values."""
    if value is None or value == "None" or value == "-":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Safely convert string to int, returning None for invalid values."""
    if value is None or value == "None" or value == "-":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _compute_growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Compute growth rate between two values."""
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)


def get_financial_metrics_av(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from Alpha Vantage."""
    cache_key = f"av_{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # Fetch all required data
    overview = _get_overview(ticker)
    income_data = _get_income_statement(ticker)
    balance_data = _get_balance_sheet(ticker)
    cash_flow_data = _get_cash_flow(ticker)

    if not overview:
        return []

    # Get financial statements based on period
    if period == "annual":
        income_reports = income_data.get("annualReports", [])
        balance_reports = balance_data.get("annualReports", [])
        cash_flow_reports = cash_flow_data.get("annualReports", [])
    else:  # ttm or quarterly
        income_reports = income_data.get("quarterlyReports", [])
        balance_reports = balance_data.get("quarterlyReports", [])
        cash_flow_reports = cash_flow_data.get("quarterlyReports", [])

    # Filter by end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    def filter_by_date(reports):
        filtered = []
        for r in reports:
            fiscal_date = r.get("fiscalDateEnding", "")
            if fiscal_date:
                try:
                    report_dt = datetime.strptime(fiscal_date, "%Y-%m-%d")
                    if report_dt <= end_dt:
                        filtered.append(r)
                except ValueError:
                    continue
        return filtered[:limit]

    income_reports = filter_by_date(income_reports)
    balance_reports = filter_by_date(balance_reports)
    cash_flow_reports = filter_by_date(cash_flow_reports)

    if not income_reports:
        # Return just overview data if no statements available
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency=overview.get("Currency", "USD"),
            market_cap=_safe_float(overview.get("MarketCapitalization")),
            enterprise_value=None,
            price_to_earnings_ratio=_safe_float(overview.get("PERatio")),
            price_to_book_ratio=_safe_float(overview.get("PriceToBookRatio")),
            price_to_sales_ratio=_safe_float(overview.get("PriceToSalesRatioTTM")),
            enterprise_value_to_ebitda_ratio=_safe_float(overview.get("EVToEBITDA")),
            enterprise_value_to_revenue_ratio=_safe_float(overview.get("EVToRevenue")),
            free_cash_flow_yield=None,
            peg_ratio=_safe_float(overview.get("PEGRatio")),
            gross_margin=_safe_float(overview.get("GrossProfitTTM")),
            operating_margin=_safe_float(overview.get("OperatingMarginTTM")),
            net_margin=_safe_float(overview.get("ProfitMargin")),
            return_on_equity=_safe_float(overview.get("ReturnOnEquityTTM")),
            return_on_assets=_safe_float(overview.get("ReturnOnAssetsTTM")),
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=None,
            quick_ratio=None,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=None,
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=_safe_float(overview.get("QuarterlyRevenueGrowthYOY")),
            earnings_growth=_safe_float(overview.get("QuarterlyEarningsGrowthYOY")),
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=_safe_float(overview.get("PayoutRatio")),
            earnings_per_share=_safe_float(overview.get("EPS")),
            book_value_per_share=_safe_float(overview.get("BookValue")),
            free_cash_flow_per_share=None,
        )
        result = [metrics]
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
        return result

    # Build metrics for each period
    metrics_list = []

    for i, income in enumerate(income_reports):
        fiscal_date = income.get("fiscalDateEnding", end_date)

        # Get corresponding balance sheet and cash flow
        balance = balance_reports[i] if i < len(balance_reports) else {}
        cash_flow = cash_flow_reports[i] if i < len(cash_flow_reports) else {}

        # Get previous period for growth calculations
        prev_income = income_reports[i + 1] if i + 1 < len(income_reports) else {}
        prev_balance = balance_reports[i + 1] if i + 1 < len(balance_reports) else {}

        # Extract values
        revenue = _safe_float(income.get("totalRevenue"))
        net_income = _safe_float(income.get("netIncome"))
        gross_profit = _safe_float(income.get("grossProfit"))
        operating_income = _safe_float(income.get("operatingIncome"))
        ebitda = _safe_float(income.get("ebitda"))

        total_assets = _safe_float(balance.get("totalAssets"))
        total_liabilities = _safe_float(balance.get("totalLiabilities"))
        current_assets = _safe_float(balance.get("totalCurrentAssets"))
        current_liabilities = _safe_float(balance.get("totalCurrentLiabilities"))
        total_equity = _safe_float(balance.get("totalShareholderEquity"))
        total_debt = _safe_float(balance.get("shortLongTermDebtTotal"))
        cash = _safe_float(balance.get("cashAndCashEquivalentsAtCarryingValue"))
        shares = _safe_float(balance.get("commonStockSharesOutstanding"))

        operating_cash_flow = _safe_float(cash_flow.get("operatingCashflow"))
        capex = _safe_float(cash_flow.get("capitalExpenditures"))

        # Previous period values
        prev_revenue = _safe_float(prev_income.get("totalRevenue"))
        prev_net_income = _safe_float(prev_income.get("netIncome"))
        prev_book_value = _safe_float(prev_balance.get("totalShareholderEquity"))

        # Compute derived metrics
        gross_margin = (gross_profit / revenue) if (gross_profit and revenue) else None
        operating_margin = (operating_income / revenue) if (operating_income and revenue) else None
        net_margin = (net_income / revenue) if (net_income and revenue) else None

        roe = (net_income / total_equity) if (net_income and total_equity) else None
        roa = (net_income / total_assets) if (net_income and total_assets) else None

        current_ratio = (current_assets / current_liabilities) if (current_assets and current_liabilities) else None
        debt_to_equity = (total_debt / total_equity) if (total_debt and total_equity) else None

        eps = (net_income / shares) if (net_income and shares) else None
        book_value_per_share = (total_equity / shares) if (total_equity and shares) else None

        fcf = None
        if operating_cash_flow is not None and capex is not None:
            fcf = operating_cash_flow - abs(capex)
        fcf_per_share = (fcf / shares) if (fcf and shares) else None

        # Growth metrics
        revenue_growth = _compute_growth(revenue, prev_revenue)
        earnings_growth = _compute_growth(net_income, prev_net_income)
        book_value_growth = _compute_growth(total_equity, prev_book_value)

        # Use overview data for market-based metrics (these are current values)
        market_cap = _safe_float(overview.get("MarketCapitalization"))
        pe_ratio = _safe_float(overview.get("PERatio"))
        pb_ratio = _safe_float(overview.get("PriceToBookRatio"))
        ps_ratio = _safe_float(overview.get("PriceToSalesRatioTTM"))
        peg_ratio = _safe_float(overview.get("PEGRatio"))
        ev_to_ebitda = _safe_float(overview.get("EVToEBITDA"))
        ev_to_revenue = _safe_float(overview.get("EVToRevenue"))

        # Compute enterprise value if we have the components
        enterprise_value = None
        if market_cap and total_debt is not None and cash is not None:
            enterprise_value = market_cap + (total_debt or 0) - (cash or 0)

        # Compute additional metrics
        debt_to_assets = (total_debt / total_assets) if (total_debt and total_assets) else None
        interest_expense = _safe_float(income.get("interestExpense"))
        interest_coverage = (operating_income / interest_expense) if (operating_income and interest_expense and interest_expense != 0) else None

        # Compute previous period values for growth
        prev_operating_income = _safe_float(prev_income.get("operatingIncome"))
        prev_ebitda = _safe_float(prev_income.get("ebitda"))
        prev_fcf = None
        if i + 1 < len(cash_flow_reports):
            prev_ocf = _safe_float(cash_flow_reports[i + 1].get("operatingCashflow"))
            prev_capex = _safe_float(cash_flow_reports[i + 1].get("capitalExpenditures"))
            if prev_ocf is not None and prev_capex is not None:
                prev_fcf = prev_ocf - abs(prev_capex)

        eps_growth = _compute_growth(eps, _safe_float(prev_income.get("netIncome")) / shares if (prev_income.get("netIncome") and shares) else None)
        fcf_growth = _compute_growth(fcf, prev_fcf)
        operating_income_growth = _compute_growth(operating_income, prev_operating_income)
        ebitda_growth = _compute_growth(ebitda, prev_ebitda)

        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=fiscal_date,
            period=period,
            currency=overview.get("Currency", "USD"),
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            price_to_earnings_ratio=pe_ratio,
            price_to_book_ratio=pb_ratio,
            price_to_sales_ratio=ps_ratio,
            enterprise_value_to_ebitda_ratio=ev_to_ebitda,
            enterprise_value_to_revenue_ratio=ev_to_revenue,
            free_cash_flow_yield=None,
            peg_ratio=peg_ratio,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=current_ratio,
            quick_ratio=None,
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            book_value_growth=book_value_growth,
            earnings_per_share_growth=eps_growth,
            free_cash_flow_growth=fcf_growth,
            operating_income_growth=operating_income_growth,
            ebitda_growth=ebitda_growth,
            payout_ratio=None,
            earnings_per_share=eps,
            book_value_per_share=book_value_per_share,
            free_cash_flow_per_share=fcf_per_share,
        )
        metrics_list.append(metrics)

    if metrics_list:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics_list])

    return metrics_list[:limit]


# Mapping of common line item names to Alpha Vantage fields
_LINE_ITEM_MAPPING = {
    # Income Statement
    "revenue": ("income", "totalRevenue"),
    "total_revenue": ("income", "totalRevenue"),
    "net_income": ("income", "netIncome"),
    "gross_profit": ("income", "grossProfit"),
    "operating_income": ("income", "operatingIncome"),
    "ebit": ("income", "ebit"),
    "ebitda": ("income", "ebitda"),
    "interest_expense": ("income", "interestExpense"),
    "income_tax_expense": ("income", "incomeTaxExpense"),
    "operating_expenses": ("income", "operatingExpenses"),
    "cost_of_revenue": ("income", "costOfRevenue"),
    "research_and_development": ("income", "researchAndDevelopment"),

    # Balance Sheet
    "total_assets": ("balance", "totalAssets"),
    "total_liabilities": ("balance", "totalLiabilities"),
    "current_assets": ("balance", "totalCurrentAssets"),
    "current_liabilities": ("balance", "totalCurrentLiabilities"),
    "shareholders_equity": ("balance", "totalShareholderEquity"),
    "total_equity": ("balance", "totalShareholderEquity"),
    "cash_and_equivalents": ("balance", "cashAndCashEquivalentsAtCarryingValue"),
    "cash_and_cash_equivalents": ("balance", "cashAndCashEquivalentsAtCarryingValue"),
    "total_debt": ("balance", "shortLongTermDebtTotal"),
    "short_term_debt": ("balance", "shortTermDebt"),
    "long_term_debt": ("balance", "longTermDebt"),
    "outstanding_shares": ("balance", "commonStockSharesOutstanding"),
    "shares_outstanding": ("balance", "commonStockSharesOutstanding"),
    "inventory": ("balance", "inventory"),
    "accounts_receivable": ("balance", "currentNetReceivables"),
    "accounts_payable": ("balance", "currentAccountsPayable"),
    "retained_earnings": ("balance", "retainedEarnings"),
    "book_value_per_share": ("balance", "bookValuePerShare"),  # computed
    "earnings_per_share": ("income", "earningsPerShare"),  # computed

    # Cash Flow
    "operating_cash_flow": ("cashflow", "operatingCashflow"),
    "capital_expenditure": ("cashflow", "capitalExpenditures"),
    "capital_expenditures": ("cashflow", "capitalExpenditures"),
    "free_cash_flow": ("cashflow", "freeCashFlow"),  # computed
    "depreciation_and_amortization": ("cashflow", "depreciationDepletionAndAmortization"),
    "dividends_and_other_cash_distributions": ("cashflow", "dividendPayout"),
    "dividend_payout": ("cashflow", "dividendPayout"),
    "issuance_or_purchase_of_equity_shares": ("cashflow", "proceedsFromRepurchaseOfEquity"),
    "net_change_in_cash": ("cashflow", "changeInCashAndCashEquivalents"),

    # Working capital (computed)
    "working_capital": ("balance", "workingCapital"),  # computed
}


def search_line_items_av(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch specific line items from Alpha Vantage financial statements."""
    # Fetch all statements
    income_data = _get_income_statement(ticker)
    balance_data = _get_balance_sheet(ticker)
    cash_flow_data = _get_cash_flow(ticker)

    # Get reports based on period
    if period == "annual":
        income_reports = income_data.get("annualReports", [])
        balance_reports = balance_data.get("annualReports", [])
        cash_flow_reports = cash_flow_data.get("annualReports", [])
    else:
        income_reports = income_data.get("quarterlyReports", [])
        balance_reports = balance_data.get("quarterlyReports", [])
        cash_flow_reports = cash_flow_data.get("quarterlyReports", [])

    # Filter by end_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    def filter_by_date(reports):
        filtered = []
        for r in reports:
            fiscal_date = r.get("fiscalDateEnding", "")
            if fiscal_date:
                try:
                    report_dt = datetime.strptime(fiscal_date, "%Y-%m-%d")
                    if report_dt <= end_dt:
                        filtered.append(r)
                except ValueError:
                    continue
        return filtered[:limit]

    income_reports = filter_by_date(income_reports)
    balance_reports = filter_by_date(balance_reports)
    cash_flow_reports = filter_by_date(cash_flow_reports)

    results = []

    # Process each period
    for i in range(min(limit, len(income_reports))):
        income = income_reports[i] if i < len(income_reports) else {}
        balance = balance_reports[i] if i < len(balance_reports) else {}
        cash_flow = cash_flow_reports[i] if i < len(cash_flow_reports) else {}

        fiscal_date = income.get("fiscalDateEnding") or balance.get("fiscalDateEnding") or end_date
        currency = income.get("reportedCurrency") or balance.get("reportedCurrency") or "USD"

        # Create LineItem with requested fields
        line_item_data = {
            "ticker": ticker,
            "report_period": fiscal_date,
            "period": period,
            "currency": currency,
        }

        statements = {
            "income": income,
            "balance": balance,
            "cashflow": cash_flow,
        }

        for item_name in line_items:
            normalized_name = item_name.lower().replace(" ", "_").replace("-", "_")

            if normalized_name in _LINE_ITEM_MAPPING:
                statement_type, field_name = _LINE_ITEM_MAPPING[normalized_name]

                # Handle computed fields
                if normalized_name == "free_cash_flow":
                    ocf = _safe_float(cash_flow.get("operatingCashflow"))
                    capex = _safe_float(cash_flow.get("capitalExpenditures"))
                    if ocf is not None and capex is not None:
                        line_item_data[item_name] = ocf - abs(capex)
                    else:
                        line_item_data[item_name] = None
                elif normalized_name == "working_capital":
                    ca = _safe_float(balance.get("totalCurrentAssets"))
                    cl = _safe_float(balance.get("totalCurrentLiabilities"))
                    if ca is not None and cl is not None:
                        line_item_data[item_name] = ca - cl
                    else:
                        line_item_data[item_name] = None
                elif normalized_name == "book_value_per_share":
                    equity = _safe_float(balance.get("totalShareholderEquity"))
                    shares = _safe_float(balance.get("commonStockSharesOutstanding"))
                    if equity is not None and shares is not None and shares > 0:
                        line_item_data[item_name] = equity / shares
                    else:
                        line_item_data[item_name] = None
                elif normalized_name == "earnings_per_share":
                    net_income = _safe_float(income.get("netIncome"))
                    shares = _safe_float(balance.get("commonStockSharesOutstanding"))
                    if net_income is not None and shares is not None and shares > 0:
                        line_item_data[item_name] = net_income / shares
                    else:
                        line_item_data[item_name] = None
                else:
                    value = statements[statement_type].get(field_name)
                    line_item_data[item_name] = _safe_float(value)
            else:
                # Try to find directly in statements
                for statement in statements.values():
                    if item_name in statement:
                        line_item_data[item_name] = _safe_float(statement[item_name])
                        break
                else:
                    line_item_data[item_name] = None

        results.append(LineItem(**line_item_data))

    return results


def get_company_news_av(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from Alpha Vantage NEWS_SENTIMENT."""
    cache_key = f"av_news_{ticker}_{start_date or 'none'}_{end_date}_{limit}"

    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": min(limit, 1000),  # Alpha Vantage max is 1000
    }

    # Note: Alpha Vantage time filtering can be unreliable
    # We fetch recent news and filter client-side if needed

    data = _make_av_request(params)
    if not data or "feed" not in data:
        return []

    # Parse date filters for client-side filtering
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None

    news_list = []
    for item in data["feed"]:
        # Parse the time_published field (format: YYYYMMDDTHHMMSS)
        time_published = item.get("time_published", "")
        if len(time_published) >= 8:
            date_str = f"{time_published[:4]}-{time_published[4:6]}-{time_published[6:8]}"
            try:
                article_dt = datetime.strptime(date_str, "%Y-%m-%d")
                # Filter by date range
                if end_dt and article_dt > end_dt:
                    continue
                if start_dt and article_dt < start_dt:
                    continue
            except ValueError:
                pass
        else:
            date_str = end_date

        # Get ticker-specific sentiment if available
        sentiment = None
        ticker_sentiments = item.get("ticker_sentiment", [])
        for ts in ticker_sentiments:
            if ts.get("ticker") == ticker:
                sentiment_score = _safe_float(ts.get("ticker_sentiment_score"))
                if sentiment_score is not None:
                    if sentiment_score > 0.2:
                        sentiment = "positive"
                    elif sentiment_score < -0.2:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                break

        # Fallback to overall sentiment
        if sentiment is None:
            overall_score = _safe_float(item.get("overall_sentiment_score"))
            if overall_score is not None:
                if overall_score > 0.2:
                    sentiment = "positive"
                elif overall_score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

        news = CompanyNews(
            ticker=ticker,
            title=item.get("title", ""),
            author=", ".join(item.get("authors", [])) or "Unknown",
            source=item.get("source", ""),
            date=date_str,
            url=item.get("url", ""),
            sentiment=sentiment,
        )
        news_list.append(news)

        # Stop if we have enough
        if len(news_list) >= limit:
            break

    if news_list:
        _cache.set_company_news(cache_key, [n.model_dump() for n in news_list])

    return news_list


def get_market_cap_av(ticker: str, end_date: str) -> Optional[float]:
    """Fetch current market cap from Alpha Vantage OVERVIEW."""
    overview = _get_overview(ticker)
    if not overview:
        return None

    return _safe_float(overview.get("MarketCapitalization"))
