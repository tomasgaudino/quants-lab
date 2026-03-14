"""
Simple Deposit/Withdrawal Detector

Detects deposits and withdrawals by looking at unexplained changes in total portfolio value.
"""

import pandas as pd
from datetime import datetime


def detect_deposits_withdrawals_simple(token_states: pd.DataFrame, threshold_pct: float = 5.0) -> pd.DataFrame:
    """
    Detect deposits/withdrawals by analyzing total portfolio value changes.

    Args:
        token_states: DataFrame with token holdings snapshots
        threshold_pct: Minimum percentage change to flag as deposit/withdrawal

    Returns:
        DataFrame with detected transactions
    """
    print(f"\n🔍 Detecting deposits/withdrawals (threshold: {threshold_pct}% change)...")

    # Group by account_state_id and calculate total value
    portfolio_values = token_states.groupby('account_state_id')['value'].sum().reset_index()
    portfolio_values.columns = ['account_state_id', 'total_value_usdt']
    portfolio_values = portfolio_values.sort_values('account_state_id')

    # Calculate changes
    portfolio_values['prev_value'] = portfolio_values['total_value_usdt'].shift(1)
    portfolio_values['value_change'] = portfolio_values['total_value_usdt'] - portfolio_values['prev_value']
    portfolio_values['pct_change'] = (portfolio_values['value_change'] / portfolio_values['prev_value'] * 100)

    # Filter for significant changes
    # A deposit/withdrawal would show as a sudden jump not explained by gradual trading
    significant_changes = portfolio_values[
        (portfolio_values['pct_change'].abs() >= threshold_pct) &
        (portfolio_values['value_change'].abs() >= 100)  # At least $100 USDT
    ].copy()

    transactions = []

    for _, row in significant_changes.iterrows():
        tx_type = "DEPOSIT" if row['value_change'] > 0 else "WITHDRAWAL"

        # Try to identify which token changed
        state_id = row['account_state_id']
        prev_state_id = row['account_state_id'] - 1

        curr_state = token_states[token_states['account_state_id'] == state_id]
        prev_state = token_states[token_states['account_state_id'] == prev_state_id] if prev_state_id >= token_states['account_state_id'].min() else pd.DataFrame()

        # Find the token with the biggest change
        token_with_max_change = None
        max_change_value = 0

        for _, curr_token in curr_state.iterrows():
            token = curr_token['token']
            curr_value = curr_token['value']

            if len(prev_state) > 0:
                prev_token = prev_state[prev_state['token'] == token]
                prev_value = prev_token['value'].iloc[0] if len(prev_token) > 0 else 0
            else:
                prev_value = 0

            change = abs(curr_value - prev_value)

            if change > max_change_value:
                max_change_value = change
                token_with_max_change = token

        transactions.append({
            'account_state_id': state_id,
            'type': tx_type,
            'value_usdt': abs(row['value_change']),
            'pct_change': row['pct_change'],
            'token_affected': token_with_max_change,
            'prev_portfolio_value': row['prev_value'],
            'curr_portfolio_value': row['total_value_usdt']
        })

    df = pd.DataFrame(transactions)

    if len(df) > 0:
        print(f"  ✓ Found {len(df)} potential deposits/withdrawals:")
        for _, row in df.iterrows():
            print(f"    {row['type']}: ${row['value_usdt']:.2f} USDT "
                  f"({row['pct_change']:+.1f}%) at state {row['account_state_id']} "
                  f"[{row['token_affected']}]")
    else:
        print("  ✓ No significant deposits/withdrawals detected")

    return df
