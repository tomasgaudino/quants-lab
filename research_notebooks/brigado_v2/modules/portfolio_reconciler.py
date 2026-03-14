"""
Portfolio Reconciliation Module

Implements flow vs stock reconciliation:
- Initial State (T0): Earliest token_states snapshot
- Event Sourcing: Chronological trade processing
- Reconciliation: Compare calculated vs actual snapshots
- Delta Tracking: Unaccounted flows (deposits, withdrawals, fees, funding)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PortfolioReconciler:
    """
    Reconciles trade events with portfolio snapshots.

    Flow (Trades) + Initial State (T0) = Stock (Snapshots) + Delta
    """

    def __init__(self, trades_df: pd.DataFrame, token_states_df: pd.DataFrame):
        """
        Initialize reconciler.

        Args:
            trades_df: Consolidated trades with timestamp, symbol, trade_type, amount, price, fee
            token_states_df: Token states with timestamp, token, units, price, value
        """
        self.trades = trades_df.copy()
        self.snapshots = token_states_df.copy()

        # Ensure timestamps are datetime and timezone-naive for comparison
        if 'timestamp' in self.trades.columns:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
            if hasattr(self.trades['timestamp'].dtype, 'tz') and self.trades['timestamp'].dtype.tz is not None:
                self.trades['timestamp'] = self.trades['timestamp'].dt.tz_localize(None)

        if 'timestamp' in self.snapshots.columns:
            self.snapshots['timestamp'] = pd.to_datetime(self.snapshots['timestamp'])
            if hasattr(self.snapshots['timestamp'].dtype, 'tz') and self.snapshots['timestamp'].dtype.tz is not None:
                self.snapshots['timestamp'] = self.snapshots['timestamp'].dt.tz_localize(None)

        # Sort by timestamp
        self.trades = self.trades.sort_values('timestamp').reset_index(drop=True)
        self.snapshots = self.snapshots.sort_values('timestamp').reset_index(drop=True)

        # Initialize state tracking
        self.initial_state = None
        self.calculated_balances = []
        self.reconciliation_deltas = []

    def get_initial_state(self) -> Dict[str, Dict]:
        """
        Get initial state from earliest token_states snapshot before first trade.

        If no snapshot exists before the first trade, calculates TRUE T0 by
        reverse-engineering the earliest snapshot using trades that occurred before it.

        Returns:
            Dict with initial balance and WAC for each asset
        """
        if len(self.trades) == 0:
            logger.warning("No trades found")
            return {}

        if len(self.snapshots) == 0:
            logger.warning("No snapshots found")
            return {}

        first_trade_time = self.trades['timestamp'].min()

        # Get earliest snapshot before or at first trade
        before_first_trade = self.snapshots[self.snapshots['timestamp'] <= first_trade_time]

        if len(before_first_trade) == 0:
            # No snapshot before trades - need to calculate TRUE T0
            logger.warning(f"No snapshot before first trade ({first_trade_time})")

            # Use earliest available snapshot
            earliest_snapshot_time = self.snapshots['timestamp'].min()
            logger.info(f"Earliest snapshot at {earliest_snapshot_time}")

            # Get trades that happened before this snapshot
            trades_before_snapshot = self.trades[self.trades['timestamp'] < earliest_snapshot_time]
            logger.warning(f"Found {len(trades_before_snapshot)} trades before earliest snapshot")

            # Get snapshot values
            initial_tokens_raw = self.snapshots[self.snapshots['timestamp'] == earliest_snapshot_time]

            # Calculate TRUE T0 by reversing the effect of trades before snapshot
            initial_state = {
                't0_timestamp': first_trade_time,  # Use first trade time as effective T0
                'assets': {},
                'calculated_from_snapshot': True
            }

            # Start with snapshot values
            snapshot_balances = {}
            for _, token_row in initial_tokens_raw.iterrows():
                token = token_row['token']
                snapshot_balances[token] = {
                    'balance': float(token_row['units']),
                    'price': float(token_row['price'])
                }

            # Reverse each trade before snapshot (in reverse chronological order)
            for _, trade in trades_before_snapshot.sort_values('timestamp', ascending=False).iterrows():
                symbol = trade['symbol']
                trade_type = trade['trade_type']
                amount = float(trade['amount'])
                price = float(trade['price'])
                fee_quote = float(trade.get('trade_fee_in_quote', 0))

                if '-' not in symbol:
                    continue

                base_asset, quote_asset = symbol.split('-')

                # Initialize if not present
                for asset in [base_asset, quote_asset]:
                    if asset not in snapshot_balances:
                        snapshot_balances[asset] = {'balance': 0.0, 'price': 1.0 if asset == 'BRL' else price}

                # Reverse the trade effect
                if trade_type == 'BUY':
                    # Reverse BUY: remove base, add back quote
                    snapshot_balances[base_asset]['balance'] -= amount
                    snapshot_balances[quote_asset]['balance'] += (price * amount + fee_quote)
                elif trade_type == 'SELL':
                    # Reverse SELL: add back base, remove quote
                    snapshot_balances[base_asset]['balance'] += amount
                    snapshot_balances[quote_asset]['balance'] -= (price * amount - fee_quote)

            # Set TRUE T0 balances
            for asset, data in snapshot_balances.items():
                initial_state['assets'][asset] = {
                    'balance': data['balance'],
                    'wac': data['price'],
                    'last_price': data['price'],
                    'source': 'calculated_from_snapshot'
                }

            initial_time = first_trade_time
            logger.warning(f"Calculated TRUE T0 by reversing {len(trades_before_snapshot)} trades")
        else:
            # Use latest snapshot before first trade
            initial_snapshot_group = before_first_trade.sort_values('timestamp').iloc[-1]
            initial_time = initial_snapshot_group['timestamp']

            # Get all tokens at initial time
            initial_tokens = self.snapshots[self.snapshots['timestamp'] == initial_time]

            initial_state = {
                't0_timestamp': initial_time,
                'assets': {}
            }

            for _, token_row in initial_tokens.iterrows():
                token = token_row['token']
                units = float(token_row['units'])
                price = float(token_row['price'])

                # Initial WAC = mark price at T0
                initial_state['assets'][token] = {
                    'balance': units,
                    'wac': price,
                    'last_price': price,
                    'source': 'initial_snapshot'
                }

        logger.info(f"Initial state (T0) at {initial_time}:")
        for token, data in initial_state['assets'].items():
            logger.info(f"  {token}: {data['balance']:.8f} @ WAC {data['wac']:.4f} (source: {data['source']})")

        self.initial_state = initial_state
        return initial_state

    def calculate_balance_evolution(self) -> pd.DataFrame:
        """
        Calculate balance evolution through event sourcing.

        Processes trades chronologically:
        - BUY: Add to inventory, update WAC
        - SELL: Reduce inventory at current WAC

        Returns:
            DataFrame with calculated balances after each trade
        """
        if self.initial_state is None:
            self.get_initial_state()

        # Initialize current state from T0
        current_balances = {
            asset: {
                'balance': data['balance'],
                'wac': data['wac'],
                'last_price': data['last_price']
            }
            for asset, data in self.initial_state['assets'].items()
        }

        # Track balance evolution
        balance_records = []

        # Add initial state record
        for asset, data in current_balances.items():
            balance_records.append({
                'timestamp': self.initial_state['t0_timestamp'],
                'event': 'initial_state',
                'asset': asset,
                'balance': data['balance'],
                'wac': data['wac'],
                'price': data['last_price'],
                'trade_id': None
            })

        # Process each trade chronologically
        for idx, trade in self.trades.iterrows():
            timestamp = trade['timestamp']
            symbol = trade['symbol']  # e.g., "BTC-BRL"
            trade_type = trade['trade_type']
            amount = float(trade['amount'])
            price = float(trade['price'])
            fee_quote = float(trade['trade_fee_in_quote']) if 'trade_fee_in_quote' in trade else 0.0

            # Parse symbol into base and quote assets
            if '-' in symbol:
                base_asset, quote_asset = symbol.split('-')
            else:
                logger.warning(f"Cannot parse symbol {symbol}, skipping")
                continue

            # Initialize assets if not present
            for asset in [base_asset, quote_asset]:
                if asset not in current_balances:
                    current_balances[asset] = {
                        'balance': 0.0,
                        'wac': 0.0,
                        'last_price': 0.0
                    }

            # Update balances based on trade type
            if trade_type == 'BUY':
                # BUY: base increases, quote decreases
                # Base asset
                old_base_balance = current_balances[base_asset]['balance']
                old_base_wac = current_balances[base_asset]['wac']

                # Update WAC for base
                if old_base_balance > 0:
                    total_cost = old_base_balance * old_base_wac
                    new_cost = price * amount + fee_quote
                    new_balance = old_base_balance + amount
                    new_wac = (total_cost + new_cost) / new_balance
                else:
                    new_balance = amount
                    new_wac = (price * amount + fee_quote) / amount if amount > 0 else price

                current_balances[base_asset]['balance'] = new_balance
                current_balances[base_asset]['wac'] = new_wac
                current_balances[base_asset]['last_price'] = price

                # Quote asset decreases
                quote_cost = price * amount + fee_quote
                current_balances[quote_asset]['balance'] -= quote_cost
                current_balances[quote_asset]['last_price'] = 1.0  # Quote is the reference

            elif trade_type == 'SELL':
                # SELL: base decreases, quote increases
                # Base asset
                current_balances[base_asset]['balance'] -= amount
                current_balances[base_asset]['last_price'] = price

                # Quote asset increases
                quote_proceeds = price * amount - fee_quote
                current_balances[quote_asset]['balance'] += quote_proceeds
                current_balances[quote_asset]['last_price'] = 1.0

            # Record state after trade
            for asset, data in current_balances.items():
                balance_records.append({
                    'timestamp': timestamp,
                    'event': 'trade',
                    'trade_type': trade_type,
                    'symbol': symbol,
                    'asset': asset,
                    'balance': data['balance'],
                    'wac': data['wac'],
                    'price': data['last_price'],
                    'trade_id': trade.get('exchange_trade_id', idx)
                })

        balance_df = pd.DataFrame(balance_records)
        self.calculated_balances = balance_df

        logger.info(f"Calculated balance evolution: {len(balance_df)} records")

        return balance_df

    def reconcile_with_snapshots(self, frequency: str = 'D') -> pd.DataFrame:
        """
        Reconcile calculated balances with actual snapshots.

        Args:
            frequency: Reconciliation frequency ('D' for daily, 'H' for hourly)

        Returns:
            DataFrame with reconciliation deltas
        """
        if self.calculated_balances is None or len(self.calculated_balances) == 0:
            self.calculate_balance_evolution()

        # Get end-of-period calculated balances
        calc_df = self.calculated_balances.copy()
        calc_df['date'] = calc_df['timestamp'].dt.date

        # Latest calculated balance per asset per day
        calc_daily = (
            calc_df.groupby(['date', 'asset'])
            .last()
            .reset_index()
        )

        # Get actual snapshots per day
        snap_df = self.snapshots.copy()
        snap_df['date'] = snap_df['timestamp'].dt.date

        # Latest snapshot per token per day
        snap_daily = (
            snap_df.groupby(['date', 'token'])
            .last()
            .reset_index()
        )

        # Merge calculated and actual
        reconciliation = calc_daily.merge(
            snap_daily,
            left_on=['date', 'asset'],
            right_on=['date', 'token'],
            how='outer',
            suffixes=('_calc', '_actual')
        )

        # Calculate deltas
        reconciliation['balance_calc'] = reconciliation['balance'].fillna(0) if 'balance' in reconciliation.columns else 0
        reconciliation['balance_actual'] = reconciliation['units'].fillna(0) if 'units' in reconciliation.columns else 0
        reconciliation['balance_delta'] = reconciliation['balance_actual'] - reconciliation['balance_calc']

        # Calculate value delta
        # Handle missing price columns
        if 'price' in reconciliation.columns and 'price_calc' in reconciliation.columns:
            reconciliation['price_actual'] = reconciliation['price'].fillna(reconciliation['price_calc'])
        elif 'price' in reconciliation.columns:
            reconciliation['price_actual'] = reconciliation['price']
        elif 'price_calc' in reconciliation.columns:
            reconciliation['price_actual'] = reconciliation['price_calc']
        else:
            reconciliation['price_actual'] = 1.0

        reconciliation['value_delta_brl'] = reconciliation['balance_delta'] * reconciliation['price_actual']

        # Clean up columns
        reconciliation = reconciliation[[
            'date',
            'asset',
            'balance_calc',
            'balance_actual',
            'balance_delta',
            'wac',
            'price_actual',
            'value_delta_brl'
        ]].sort_values(['date', 'asset'])

        # Log significant deltas
        significant = reconciliation[abs(reconciliation['value_delta_brl']) > 1.0]
        if len(significant) > 0:
            logger.warning(f"Found {len(significant)} significant deltas (>R$1):")
            for _, row in significant.iterrows():
                logger.warning(
                    f"  {row['date']} {row['asset']}: "
                    f"Delta={row['balance_delta']:.8f} "
                    f"(R$ {row['value_delta_brl']:.2f})"
                )

        self.reconciliation_deltas = reconciliation

        logger.info(f"Reconciliation complete: {len(reconciliation)} records")

        return reconciliation

    def get_summary(self) -> Dict:
        """
        Get reconciliation summary statistics.

        Returns:
            Dict with summary metrics
        """
        if self.reconciliation_deltas is None or len(self.reconciliation_deltas) == 0:
            self.reconcile_with_snapshots()

        total_delta_brl = self.reconciliation_deltas['value_delta_brl'].sum()

        summary = {
            'initial_state': self.initial_state,
            'total_trades': len(self.trades),
            'total_snapshots': len(self.snapshots),
            'total_delta_brl': total_delta_brl,
            'abs_total_delta_brl': abs(total_delta_brl),
            'avg_daily_delta_brl': total_delta_brl / self.reconciliation_deltas['date'].nunique() if len(self.reconciliation_deltas) > 0 else 0,
            'max_single_delta_brl': self.reconciliation_deltas['value_delta_brl'].abs().max(),
            'reconciliation_accuracy': 1.0 - (abs(total_delta_brl) / self.initial_state['assets'].get('BRL', {}).get('balance', 1000) if self.initial_state else 0)
        }

        return summary
