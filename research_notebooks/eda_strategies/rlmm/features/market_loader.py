# market_loader.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class MarketLoader:
    """
    Unified loader for candles, orderbooks, and trades data.

    Consolidates all market data loading functionality into a single interface.
    """

    def __init__(
        self,
        trading_pair: str = "USDT-BRL",
        exchange: str = "binance",
        date_range: Optional[Tuple[str, str]] = None,
    ):
        """
        Parameters
        ----------
        trading_pair : str
            Trading pair symbol, e.g. "USDT-BRL"
        exchange : str
            Exchange name, e.g. "binance"
        date_range : Optional[Tuple[str, str]]
            Date range as (start_date, end_date) in 'YYYY-MM-DD' format
        """
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.date_range = date_range

    # ==========================================
    # Candles Loading
    # ==========================================

    def load_candles(
        self,
        path: str | Path,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """
        Load candles from a Binance-style CSV.

        Expected columns:
          - timestamp (epoch seconds)
          - open, high, low, close
          - volume
          - quote_asset_volume
          - n_trades
          - taker_buy_base_volume
          - taker_buy_quote_volume

        Parameters
        ----------
        path : str | Path
            Full path to the candles CSV file
        interval : str
            Candle interval (informational), e.g. "1m"

        Returns
        -------
        pd.DataFrame
            Candles data with timestamp (UTC), OHLCV, and volume metrics
        """
        path = Path(path)

        df = pd.read_csv(path)

        if "timestamp" not in df.columns:
            raise ValueError("Candles CSV must have a 'timestamp' column")

        # timestamp viene como epoch seconds (float) -> datetime UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        # Ordenamos por tiempo
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filtro por rango de fechas si se pide
        if self.date_range is not None:
            start_date, end_date = self.date_range
            start = pd.to_datetime(start_date).tz_localize("UTC")
            # end inclusive: sumamos 1 día y usamos <
            end = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)

            mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
            df = df.loc[mask].reset_index(drop=True)

        return df

    # ==========================================
    # Orderbook Loading
    # ==========================================

    @staticmethod
    def _parse_orderbook_line(line: str) -> Optional[dict]:
        """
        Parse a single JSON line from the order book snapshot file.

        Expected format:
        {
          "ts": 1760106864.0,
          "bids": [[price, size], ...],
          "asks": [[price, size], ...]
        }
        """
        line = line.strip()
        if not line:
            return None

        data = json.loads(line)

        ts = data["ts"]
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Convert epoch seconds to UTC datetime
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Best bid = max price in bids
        if bids:
            best_bid_price, best_bid_size = max(bids, key=lambda x: x[0])
        else:
            best_bid_price, best_bid_size = np.nan, np.nan

        # Best ask = min price in asks
        if asks:
            best_ask_price, best_ask_size = min(asks, key=lambda x: x[0])
        else:
            best_ask_price, best_ask_size = np.nan, np.nan

        return {
            "timestamp": timestamp,
            "best_bid": best_bid_price,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask_price,
            "best_ask_size": best_ask_size,
        }

    def load_orderbooks(
        self,
        folder: str | Path,
    ) -> pd.DataFrame:
        """
        Load all order book snapshot files for the trading pair from a folder.

        Files are expected to have names like:
          binance_USDT-BRL_order_book_snapshots_YYYY-MM-DD.txt

        Parameters
        ----------
        folder : str | Path
            Folder containing orderbook snapshot txt files

        Returns
        -------
        pd.DataFrame
            Orderbook data with columns:
            - timestamp (UTC)
            - best_bid, best_bid_size
            - best_ask, best_ask_size
        """
        folder = Path(folder)

        pattern = f"{self.exchange}_{self.trading_pair}_order_book_snapshots_*.txt"
        files = sorted(folder.glob(pattern))

        if self.date_range is not None:
            start_date, end_date = self.date_range
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()

            def in_range(path: Path) -> bool:
                # Extract date from filename (last part before .txt)
                # binance_USDT-BRL_order_book_snapshots_2025-10-10.txt
                date_str = path.stem.split("_")[-1]
                d = pd.to_datetime(date_str).date()
                return start <= d <= end

            files = [f for f in files if in_range(f)]

        rows: List[dict] = []

        for path in files:
            with path.open("r") as f:
                for line in f:
                    parsed = self._parse_orderbook_line(line)
                    if parsed is not None:
                        rows.append(parsed)

        if not rows:
            raise ValueError(
                f"No orderbook data found in folder '{folder}' with pattern '{pattern}'"
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ==========================================
    # Trades Loading
    # ==========================================

    @staticmethod
    def _parse_trade_line(line: str) -> Optional[dict]:
        """
        Parse a single JSON line from the trades file.

        Expected format:
        {
          "ts": 1760106864.009,
          "price": 5.4485,
          "q_base": 10.5,
          "side": "buy"
        }
        """
        line = line.strip()
        if not line:
            return None

        data = json.loads(line)

        ts = data["ts"]
        price = data["price"]
        q_base = data["q_base"]
        side = data["side"]

        # Epoch seconds -> UTC datetime
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)

        return {
            "timestamp": timestamp,
            "price": float(price),
            "size": float(q_base),  # usamos 'size' como nombre estándar
            "side": str(side).lower(),
        }

    def load_trades(
        self,
        folder: str | Path,
    ) -> pd.DataFrame:
        """
        Load all trade files for the trading pair from a folder.

        Files are expected to have names like:
          binance_USDT-BRL_trades_YYYY-MM-DD.txt

        Parameters
        ----------
        folder : str | Path
            Folder containing trade txt files

        Returns
        -------
        pd.DataFrame
            Trades data with columns:
            - timestamp (UTC)
            - price
            - size  (base quantity, from q_base)
            - side  ('buy' or 'sell')
        """
        folder = Path(folder)

        pattern = f"{self.exchange}_{self.trading_pair}_trades_*.txt"
        files = sorted(folder.glob(pattern))

        if self.date_range is not None:
            start_date, end_date = self.date_range
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()

            def in_range(path: Path) -> bool:
                # binance_USDT-BRL_trades_2025-10-10.txt
                date_str = path.stem.split("_")[-1]
                d = pd.to_datetime(date_str).date()
                return start <= d <= end

            files = [f for f in files if in_range(f)]

        rows: List[dict] = []

        for path in files:
            with path.open("r") as f:
                for line in f:
                    parsed = self._parse_trade_line(line)
                    if parsed is not None:
                        rows.append(parsed)

        if not rows:
            raise ValueError(
                f"No trades data found in folder '{folder}' with pattern '{pattern}'"
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ==========================================
    # Unified Loading
    # ==========================================

    def load_all(
        self,
        candles_path: str | Path,
        orderbooks_folder: str | Path,
        trades_folder: str | Path,
        candles_interval: str = "1m",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three data sources at once.

        Parameters
        ----------
        candles_path : str | Path
            Path to candles CSV file
        orderbooks_folder : str | Path
            Folder containing orderbook snapshots
        trades_folder : str | Path
            Folder containing trades
        candles_interval : str
            Candle interval, e.g. "1m"

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (candles_df, orderbooks_df, trades_df)
        """
        candles = self.load_candles(candles_path, interval=candles_interval)
        orderbooks = self.load_orderbooks(orderbooks_folder)
        trades = self.load_trades(trades_folder)

        return candles, orderbooks, trades
