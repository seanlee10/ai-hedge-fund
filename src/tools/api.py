"""
Financial data API module.

Uses Alpha Vantage for prices, financial metrics, line items, and news.
Uses SEC EDGAR for insider trading data.
"""

import pandas as pd

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

# Import from new provider modules
from src.tools.alpha_vantage import (
    get_prices_av,
    get_financial_metrics_av,
    search_line_items_av,
    get_company_news_av,
    get_market_cap_av,
)
from src.tools.sec_edgar import get_insider_trades_sec


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        List of Price objects with OHLCV data
    """
    return get_prices_av(ticker, start_date, end_date)


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        end_date: End date for filtering reports (YYYY-MM-DD)
        period: Report period type - "ttm", "quarterly", or "annual"
        limit: Maximum number of periods to return
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        List of FinancialMetrics objects with valuation ratios, margins, etc.
    """
    return get_financial_metrics_av(ticker, end_date, period, limit)


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch specific financial statement line items from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        line_items: List of line item names to fetch (e.g., ["revenue", "net_income"])
        end_date: End date for filtering reports (YYYY-MM-DD)
        period: Report period type - "ttm", "quarterly", or "annual"
        limit: Maximum number of periods to return
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        List of LineItem objects with requested fields
    """
    return search_line_items_av(ticker, line_items, end_date, period, limit)


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from SEC EDGAR Form 4 filings.

    Args:
        ticker: Stock ticker symbol
        end_date: End date for filtering trades (YYYY-MM-DD)
        start_date: Optional start date for filtering trades (YYYY-MM-DD)
        limit: Maximum number of trades to return
        api_key: Not used (SEC EDGAR is free), kept for API compatibility

    Returns:
        List of InsiderTrade objects with transaction details
    """
    return get_insider_trades_sec(ticker, end_date, start_date, limit)


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from Alpha Vantage NEWS_SENTIMENT endpoint.

    Args:
        ticker: Stock ticker symbol
        end_date: End date for filtering news (YYYY-MM-DD)
        start_date: Optional start date for filtering news (YYYY-MM-DD)
        limit: Maximum number of news items to return
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        List of CompanyNews objects with title, date, sentiment, etc.
    """
    return get_company_news_av(ticker, end_date, start_date, limit)


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market capitalization from Alpha Vantage.

    Args:
        ticker: Stock ticker symbol
        end_date: Date for market cap (uses current value from OVERVIEW)
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        Market capitalization as float, or None if unavailable
    """
    return get_market_cap_av(ticker, end_date)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame.

    Args:
        prices: List of Price objects

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    if not prices:
        return pd.DataFrame()

    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    """Fetch price data and return as DataFrame.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: Optional API key (uses ALPHA_VANTAGE_API_KEY env var if not provided)

    Returns:
        DataFrame with Date index and OHLCV columns
    """
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
