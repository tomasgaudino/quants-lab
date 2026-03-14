"""
Price Fetcher

Fetches current market prices from exchanges for PnL calculation.
"""

import requests
from typing import Dict, Optional
from datetime import datetime


class PriceFetcher:
    """Fetch current prices from exchanges."""

    def __init__(self):
        self.cache = {}
        self.fetch_time = None

    def get_binance_price(self, symbol: str) -> Optional[float]:
        """
        Get current price from Binance.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC-BRL', 'USDT-BRL')

        Returns:
            Current price or None if fetch fails
        """
        # Convert symbol format: BTC-BRL -> BTCBRL
        binance_symbol = symbol.replace('-', '')

        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                return price
            else:
                print(f"  ⚠️  Failed to fetch {symbol} price from Binance (status: {response.status_code})")
                return None

        except Exception as e:
            print(f"  ⚠️  Error fetching {symbol} price: {e}")
            return None

    def get_current_prices(self, symbols: list, exchange: str = 'binance') -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            exchange: Exchange name (default: 'binance')

        Returns:
            Dict mapping symbol to current price
        """
        prices = {}
        self.fetch_time = datetime.now()

        print(f"\n📊 Fetching current prices from {exchange}...")

        for symbol in symbols:
            if exchange.lower() == 'binance':
                price = self.get_binance_price(symbol)
                if price:
                    prices[symbol] = price
                    print(f"  ✓ {symbol}: {price:,.2f}")
                else:
                    prices[symbol] = None
                    print(f"  ✗ {symbol}: Failed to fetch")

        self.cache = prices
        return prices

    def get_price(self, symbol: str, exchange: str = 'binance') -> Optional[float]:
        """
        Get current price for a single symbol.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            Current price or None if not available
        """
        # Check cache first
        if symbol in self.cache:
            return self.cache[symbol]

        # Fetch if not cached
        if exchange.lower() == 'binance':
            price = self.get_binance_price(symbol)
            self.cache[symbol] = price
            return price

        return None
