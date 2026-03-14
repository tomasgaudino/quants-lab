"""
Deposit/Withdrawal Detector

Detects deposits and withdrawals from token_states by identifying sudden changes
in token balances that can't be explained by trading activity.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime


class DepositWithdrawalDetector:
    """Detect deposits and withdrawals from portfolio snapshots."""

    def __init__(self, token_states: pd.DataFrame, trades: pd.DataFrame):
        """
        Initialize detector.

        Args:
            token_states: DataFrame with token holdings snapshots
            trades: DataFrame with all trades
        """
        self.token_states = token_states.copy()
        self.trades = trades.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in self.trades.columns:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])

    def detect_deposits_withdrawals(self, threshold_usdt: float = 100.0) -> pd.DataFrame:
        """
        Detect deposits and withdrawals by comparing balance changes to trade activity.

        Args:
            threshold_usdt: Minimum value change (in USDT) to consider as deposit/withdrawal

        Returns:
            DataFrame with detected deposits/withdrawals
        """
        print(f"\n🔍 Detecting deposits/withdrawals (threshold: {threshold_usdt} USDT)...")

        # Get all account states sorted by ID
        account_states = sorted(self.token_states['account_state_id'].unique())

        deposits_withdrawals = []

        for i in range(1, len(account_states)):
            prev_state_id = account_states[i - 1]
            curr_state_id = account_states[i]

            # Get states
            prev_state = self.token_states[self.token_states['account_state_id'] == prev_state_id]
            curr_state = self.token_states[self.token_states['account_state_id'] == curr_state_id]

            # Get timestamps (assuming we can derive from state IDs or trades)
            # For now, we'll use account_state_id as a proxy for time
            timestamp = self._estimate_timestamp(curr_state_id)

            # Compare each token
            all_tokens = set(prev_state['token'].unique()) | set(curr_state['token'].unique())

            for token in all_tokens:
                prev_token = prev_state[prev_state['token'] == token]
                curr_token = curr_state[curr_state['token'] == token]

                prev_units = float(prev_token['units'].iloc[0]) if len(prev_token) > 0 else 0.0
                curr_units = float(curr_token['units'].iloc[0]) if len(curr_token) > 0 else 0.0

                # Calculate expected change from trades
                expected_change = self._calculate_expected_change(
                    token, prev_state_id, curr_state_id
                )

                # Actual change
                actual_change = curr_units - prev_units

                # Unexplained change
                unexplained = actual_change - expected_change

                # Get value in USDT
                token_price_usdt = float(curr_token['price'].iloc[0]) if len(curr_token) > 0 else 0.0
                unexplained_value_usdt = abs(unexplained * token_price_usdt)

                # If significant unexplained change, it's likely a deposit/withdrawal
                if unexplained_value_usdt >= threshold_usdt:
                    tx_type = "DEPOSIT" if unexplained > 0 else "WITHDRAWAL"

                    deposits_withdrawals.append({
                        'account_state_id': curr_state_id,
                        'timestamp': timestamp,
                        'token': token,
                        'amount': abs(unexplained),
                        'type': tx_type,
                        'price_usdt': token_price_usdt,
                        'value_usdt': unexplained_value_usdt,
                        'prev_units': prev_units,
                        'curr_units': curr_units,
                        'expected_change': expected_change,
                        'actual_change': actual_change
                    })

        df = pd.DataFrame(deposits_withdrawals)

        if len(df) > 0:
            print(f"  ✓ Found {len(df)} deposits/withdrawals")
            for _, row in df.iterrows():
                print(f"    {row['type']}: {row['amount']:.4f} {row['token']} "
                      f"(${row['value_usdt']:.2f} USDT) at state {row['account_state_id']}")
        else:
            print("  ✓ No significant deposits/withdrawals detected")

        return df

    def _estimate_timestamp(self, account_state_id: int) -> datetime:
        """
        Estimate timestamp for an account state.

        Since token_states doesn't have timestamps, we estimate based on:
        1. Trades that happened around that state
        2. Or use state_id as a sequential indicator
        """
        # Try to find trades around this state ID
        # This is a rough estimation - in production you'd want actual timestamps
        # For now, we'll create a synthetic timestamp based on state ID
        # Assuming states are roughly sequential in time

        # Get first and last state IDs
        min_state = self.token_states['account_state_id'].min()
        max_state = self.token_states['account_state_id'].max()

        # Get first and last trade timestamps if available
        if len(self.trades) > 0 and 'timestamp' in self.trades.columns:
            min_time = self.trades['timestamp'].min()
            max_time = self.trades['timestamp'].max()

            # Linear interpolation
            time_range = (max_time - min_time).total_seconds()
            state_range = max_state - min_state

            if state_range > 0:
                ratio = (account_state_id - min_state) / state_range
                estimated = min_time + pd.Timedelta(seconds=time_range * ratio)
                return estimated

        # Fallback: use current time with state ID offset
        return datetime.now()

    def _calculate_expected_change(self, token: str, from_state_id: int, to_state_id: int) -> float:
        """
        Calculate expected change in token balance based on trades.

        Args:
            token: Token symbol
            from_state_id: Previous state ID
            to_state_id: Current state ID

        Returns:
            Expected change in token units
        """
        # For simplicity, we'll assume trades are roughly evenly distributed across states
        # In reality, you'd want to match trades to specific state IDs with timestamps

        # Get all trades for this token
        token_trades = self.trades[
            (self.trades['base_asset'] == token) | (self.trades['quote_asset'] == token)
        ].copy()

        if len(token_trades) == 0:
            return 0.0

        # Calculate net change from trades
        # This is a simplified approach - ideally you'd filter trades by timestamp range
        total_change = 0.0

        for _, trade in token_trades.iterrows():
            if trade['base_asset'] == token:
                # This is the base asset being traded
                if trade['trade_type'] == 'BUY':
                    total_change += trade['amount']
                else:  # SELL
                    total_change -= trade['amount']
            elif trade['quote_asset'] == token:
                # This is the quote asset
                quote_volume = trade['price'] * trade['amount']
                if trade['trade_type'] == 'BUY':
                    total_change -= quote_volume
                else:  # SELL
                    total_change += quote_volume

        # Distribute change across state range
        # This is a rough estimate - assumes uniform distribution
        min_state = self.token_states['account_state_id'].min()
        max_state = self.token_states['account_state_id'].max()

        if max_state > min_state:
            state_fraction = (to_state_id - from_state_id) / (max_state - min_state)
            return total_change * state_fraction

        return 0.0
