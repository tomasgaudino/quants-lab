from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging
import asyncio
import os
from typing import List, Dict, Any

from scipy import stats
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

from core.data_sources import CLOBDataSource
from core.services.mongodb_client import MongoDBClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class CointegrationTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoDBClient(**config.get("db_config", {}))
        self.clob = CLOBDataSource()

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            trading_rules = await self.clob.get_trading_rules(connector_name='binance_perpetual')
            trading_pairs = trading_rules.get_all_trading_pairs()
            candles_config = self.config.get("candles_config", {})
            candles_config["trading_pairs"] = trading_pairs[:50]
            candles = await self.clob.get_candles_batch_last_days(**candles_config)
            volume_medians = pd.DataFrame(columns=['trading_pair', 'usdt_volume'])

            for candle in candles:
                if candle.data is not None:
                    volume_medians.loc[len(volume_medians)] = [candle.trading_pair,
                                                               candle.data['quote_asset_volume'].sum()]

            volume_medians = volume_medians[
                (volume_medians['usdt_volume'] >= volume_medians['usdt_volume'].quantile(0.75)) & (
                    ~volume_medians['trading_pair'].str.contains("USDC"))]

            candles_dict = {candle.trading_pair: candle.data for candle in candles if
                            candle.trading_pair in volume_medians['trading_pair'].tolist()}

            cointegration_results: List[Dict[str, Any]] = self.analyze_trading_pairs(candles_dict,
                                                                                     z_score_threshold=0.5)
            logging.info(f"Successfully added {len(cointegration_results)} cointegration records")

        except Exception as e:
            logging.error(f"Error in Cointegration Task: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()

    def analyze_trading_pairs(self, candles_dict, z_score_threshold=1.5, grid_levels=5):
        """
        Analyze a dictionary of trading pairs and return a list of dictionaries with grid trading parameters

        Args:
            candles_dict: Dictionary where keys are pair names and values are candle dataframes
            z_score_threshold: Threshold for z-score signals
            grid_levels: Number of grid levels to generate

        Returns:
            List of dictionaries with grid trading parameters for all cointegrated pairs
        """
        # Get the analysis DataFrame
        results_df = self.analyze_multiple_pairs(candles_dict,
                                                 z_score_threshold=z_score_threshold,
                                                 grid_levels=grid_levels)

        # Filter for cointegrated pairs
        pair_results = []
        processed_pairs = set()

        for _, row in results_df.iterrows():
            pair_key = tuple(sorted([row['Base'], row['Quote']]))

            # Skip if we've already processed this pair
            if pair_key in processed_pairs:
                continue

            # Get both directions for this pair
            pair_analysis = results_df[
                ((results_df['Base'] == row['Base']) & (results_df['Quote'] == row['Quote'])) |
                ((results_df['Base'] == row['Quote']) & (results_df['Quote'] == row['Base']))
                ]

            # Check if both directions are cointegrated
            if not (pair_analysis['P-Value'] < 0.05).all():
                continue

            # Find long and short positions
            long_position = pair_analysis[pair_analysis['Strategy'] == 'Long'].iloc[0] if len(
                pair_analysis[pair_analysis['Strategy'] == 'Long']) > 0 else None
            short_position = pair_analysis[pair_analysis['Strategy'] == 'Short'].iloc[0] if len(
                pair_analysis[pair_analysis['Strategy'] == 'Short']) > 0 else None

            # Skip if we don't have both positions or if any grid prices are zero/None
            if (long_position is None or short_position is None or
                    not long_position['Entry_Price'] or not long_position['Target_Price'] or not long_position[
                        'Stop_Price'] or
                    not short_position['Entry_Price'] or not short_position['Target_Price'] or not short_position[
                        'Stop_Price'] or
                    long_position['Entry_Price'] == 0 or long_position['Target_Price'] == 0 or long_position[
                        'Stop_Price'] == 0 or
                    short_position['Entry_Price'] == 0 or short_position['Target_Price'] == 0 or short_position[
                        'Stop_Price'] == 0):
                continue

            # Calculate average cointegration value
            coint_value = pair_analysis['P-Value'].mean()

            # Create result dictionary for this pair
            result = {
                'base': long_position['Base'],
                'quote': short_position['Base'],
                'grid_base': {
                    'start_price': float(long_position['Entry_Price']),
                    'end_price': float(long_position['Target_Price']),
                    'limit_price': float(long_position['Stop_Price']),
                    'beta': float(long_position['Beta'])  # Replace max_open_orders with beta
                },
                'grid_quote': {
                    'start_price': float(short_position['Entry_Price']),
                    'end_price': float(short_position['Target_Price']),
                    'limit_price': float(short_position['Stop_Price']),
                    'beta': float(short_position['Beta'])  # Replace max_open_orders with beta
                },
                'coint_value': float(coint_value)
            }
            pair_results.append(result)

            # Mark this pair as processed
            processed_pairs.add(pair_key)

        # Sort results by cointegration value
        pair_results.sort(key=lambda x: x['coint_value'])

        # Print summary
        print(f"\nFound {len(pair_results)} cointegrated pairs:")
        print("-" * 50)
        for result in pair_results:
            print(f"\nLong {result['base']} vs Short {result['quote']}")
            print(f"Cointegration value: {result['coint_value']:.4f}")
            print(f"Grid Base - Entry: {result['grid_base']['start_price']:.2f}, "
                  f"Target: {result['grid_base']['end_price']:.2f}, "
                  f"Stop: {result['grid_base']['limit_price']:.2f}, "
                  f"Beta: {result['grid_base']['beta']:.4f}")
            print(f"Grid Quote - Entry: {result['grid_quote']['start_price']:.2f}, "
                  f"Target: {result['grid_quote']['end_price']:.2f}, "
                  f"Stop: {result['grid_quote']['limit_price']:.2f}, "
                  f"Beta: {result['grid_quote']['beta']:.4f}")

        return pair_results

    def analyze_multiple_pairs(self, candles_dict, z_score_threshold=1.5, grid_levels=5):
        """
        Analyze all possible combinations of trading pairs and return results in a DataFrame

        Args:
            candles_dict: Dictionary of candle data {pair_name: candle_dataframe}
            z_score_threshold: Threshold for z-score signals
            grid_levels: Number of grid levels to generate

        Returns:
            DataFrame with analysis results for all pair combinations
        """
        # Create lists to store results
        results = []
        pair_names = list(candles_dict.keys())

        # Analyze each possible pair combination
        for i in range(len(pair_names)):
            for j in range(i + 1, len(pair_names)):
                pair1 = pair_names[i]
                pair2 = pair_names[j]

                print(f"\nAnalyzing {pair1} vs {pair2}")

                try:
                    # Get candle data
                    candle1 = candles_dict[pair1]
                    candle2 = candles_dict[pair2]

                    # Get normalized price series
                    price1 = candle1["close"].pct_change().add(1).cumprod()
                    price2 = candle2["close"].pct_change().add(1).cumprod()

                    # Analyze both directions
                    analysis_1vs2 = self.analyze_pair_cointegration(price1, price2,
                                                                    lookback_days=14,
                                                                    signal_days=14,
                                                                    z_score_threshold=z_score_threshold)

                    analysis_2vs1 = self.analyze_pair_cointegration(price2, price1,
                                                                    lookback_days=14,
                                                                    signal_days=14,
                                                                    z_score_threshold=z_score_threshold)

                    # Generate grid levels
                    current_price1 = candle1["close"].iloc[-1]
                    current_price2 = candle2["close"].iloc[-1]

                    grid_1vs2 = self.generate_grid_levels(analysis_1vs2, current_price1,
                                                          entry_threshold=z_score_threshold,
                                                          grid_levels=grid_levels)

                    grid_2vs1 = self.generate_grid_levels(analysis_2vs1, current_price2,
                                                          entry_threshold=z_score_threshold,
                                                          grid_levels=grid_levels)

                    # Store results for first direction
                    results.append({
                        'Base': pair1,
                        'Quote': pair2,
                        'P-Value': analysis_1vs2['P-Value'],
                        'Z-Score': analysis_1vs2['Current_Z_score'],
                        'Strategy': analysis_1vs2['Strategy'],
                        'Signal_Strength': analysis_1vs2['Signal_Strength'],
                        'Mean_Reversion_Prob': analysis_1vs2['Mean_Reversion_Probability'],
                        'Beta': analysis_1vs2['Beta'],
                        'Entry_Price': grid_1vs2['entry_price'] if grid_1vs2['strategy'] != 'Hold' else None,
                        'Target_Price': grid_1vs2['target_price'] if grid_1vs2['strategy'] != 'Hold' else None,
                        'Stop_Price': grid_1vs2['stop_price'] if grid_1vs2['strategy'] != 'Hold' else None
                    })

                    # Store results for second direction
                    results.append({
                        'Base': pair2,
                        'Quote': pair1,
                        'P-Value': analysis_2vs1['P-Value'],
                        'Z-Score': analysis_2vs1['Current_Z_score'],
                        'Strategy': analysis_2vs1['Strategy'],
                        'Signal_Strength': analysis_2vs1['Signal_Strength'],
                        'Mean_Reversion_Prob': analysis_2vs1['Mean_Reversion_Probability'],
                        'Beta': analysis_2vs1['Beta'],
                        'Entry_Price': grid_2vs1['entry_price'] if grid_2vs1['strategy'] != 'Hold' else None,
                        'Target_Price': grid_2vs1['target_price'] if grid_2vs1['strategy'] != 'Hold' else None,
                        'Stop_Price': grid_2vs1['stop_price'] if grid_2vs1['strategy'] != 'Hold' else None
                    })

                except Exception as e:
                    print(f"Error analyzing {pair1} vs {pair2}: {str(e)}")
                    continue

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add derived columns
        df['Cointegrated'] = df['P-Value'] < 0.05
        df['Potential_Profit'] = np.where(df['Strategy'] != 'Hold',
                                          abs(df['Target_Price'] - df['Entry_Price']) / df['Entry_Price'],
                                          0)
        df['Risk_Ratio'] = np.where(df['Strategy'] != 'Hold',
                                    abs(df['Target_Price'] - df['Entry_Price']) /
                                    abs(df['Stop_Price'] - df['Entry_Price']),
                                    0)

        # Sort by signal strength and potential profit
        df = df.sort_values(['Signal_Strength', 'Potential_Profit'],
                            ascending=[False, False])

        return df

    @staticmethod
    def generate_grid_levels(analysis, current_price, entry_threshold=1.5, stop_threshold=1.0, grid_levels=5,
                             time_limit_hours=24):
        """
        Generate grid trading levels based on Z-scores with configurable thresholds.

        Args:
            analysis (dict): Results from cointegration analysis
            current_price (float): Current price of the asset
            entry_threshold (float): Z-score threshold for entry (default 1.5)
            stop_threshold (float): Additional Z-score deviation for stop loss (default 1.0)
            grid_levels (int): Number of grid levels to generate between entry and target
            time_limit_hours (int): Time limit for the grid validity

        Returns:
            dict: Grid trading parameters and levels
        """
        z_score = analysis['Current_Z_score']
        z_mean = analysis['Z_mean']
        z_std = analysis['Z_std']
        beta = analysis['Position_Ratio']

        # Time parameters
        current_time = analysis['Actual_Values'].index[-1]
        time_limit = current_time + timedelta(hours=time_limit_hours)

        if abs(z_score) > entry_threshold:
            is_short = z_score > 0

            entry_price = current_price

            # Calculate target and stop prices
            if is_short:
                target_price = current_price * (1 - (z_score * z_std * beta))
                stop_price = current_price * (1 + (stop_threshold * z_std * beta))
                grid_direction = -1
            else:  # Long
                target_price = current_price * (1 + (abs(z_score) * z_std * beta))
                stop_price = current_price * (1 - (stop_threshold * z_std * beta))
                grid_direction = 1

            # Generate grid levels
            price_range = abs(target_price - entry_price)
            grid_step = price_range / (grid_levels + 1)
            grid_prices = [entry_price + (i * grid_step * grid_direction) for i in range(1, grid_levels + 1)]

            grid = {
                'strategy': 'Short' if is_short else 'Long',
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_price': stop_price,
                'grid_prices': grid_prices,
                'time_limit': time_limit,
                'entry_z_score': z_score,
                'target_z_score': 0,
                'stop_z_score': z_score + (stop_threshold * (1 if is_short else -1)),
                'grid_levels': grid_levels,
                'grid_step': grid_step
            }

        else:  # No signal
            grid = {
                'strategy': 'Hold',
                'entry_price': None,
                'target_price': None,
                'stop_price': None,
                'grid_prices': [],
                'time_limit': None,
                'entry_z_score': z_score,
                'target_z_score': None,
                'stop_z_score': None,
                'grid_levels': 0,
                'grid_step': None
            }
        return grid

    @staticmethod
    def analyze_pair_cointegration(y_col, x_col, lookback_days=14, signal_days=3, z_score_threshold=2.0):
        """
        Comprehensive cointegration analysis combining spread analysis and trading signals.

        Args:
            y_col (pd.Series): The Y series (dependent variable)
            x_col (pd.Series): The X series (independent variable)
            lookback_days (int): Days of data to use for cointegration analysis
            signal_days (int): Recent days to analyze for trading signals
            z_score_threshold (float): Z-score threshold for trading signals

        Returns:
            dict: Comprehensive analysis results including trading signals
        """
        # Calculate periods for 15m candles
        lookback_periods = lookback_days * 24 * 4  # 15-min candles per day
        signal_periods = signal_days * 24 * 4

        # Prepare price series
        y_col = y_col.dropna()
        x_col = x_col.dropna()

        # Ensure finite values
        y_col = y_col[np.isfinite(y_col)]
        x_col = x_col[np.isfinite(x_col)]

        # Get last n periods for analysis
        y_col = y_col.tail(lookback_periods)
        x_col = x_col.tail(lookback_periods)

        # Ensure both series are of the same length
        min_len = min(len(y_col), len(x_col))
        y_col = y_col[-min_len:]
        x_col = x_col[-min_len:]

        y, x = y_col.values, x_col.values

        # Run Engle-Granger test
        coint_res = coint(y, x)
        p_value = coint_res[1]

        # Perform linear regression
        x_reshaped = x.reshape(-1, 1)
        reg = LinearRegression().fit(x_reshaped, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]

        # Calculate spread (Z_t)
        z_t = pd.Series(y - (alpha + beta * x))
        z_t.index = y_col.index.copy()
        z_mean = np.mean(z_t)
        z_std = np.std(z_t)

        # Get recent data for signal analysis
        y_recent = y_col.tail(signal_periods)
        x_recent = x_col.tail(signal_periods)

        # Calculate recent predictions and spread
        y_pred = alpha + beta * x_recent
        recent_spread = y_recent - y_pred

        # Calculate current Z-score
        current_z_score = (z_t[-1] - z_mean) / z_std

        # Determine trading strategy
        if abs(current_z_score) < z_score_threshold:
            strategy = "Hold"
        elif current_z_score > z_score_threshold:
            strategy = "Short"  # Y is overvalued
        else:  # current_z_score < -z_score_threshold
            strategy = "Long"  # Y is undervalued

        # Calculate additional metrics
        signal_strength = abs(current_z_score) / z_score_threshold
        mean_reversion_prob = 1 - stats.norm.cdf(abs(current_z_score))

        # Calculate percentage error for recent period
        percentage_error = ((y_pred - y_recent) / y_recent.abs()) * 100
        median_error = np.median(percentage_error)

        return {
            # Cointegration statistics
            'P-Value': p_value,
            'Alpha': alpha,
            'Beta': beta,

            # Spread analysis
            'Z_t': z_t,
            'Z_mean': z_mean,
            'Z_std': z_std,
            'Current_Z_score': current_z_score,

            # Trading signals
            'Strategy': strategy,
            'Signal_Strength': signal_strength,
            'Mean_Reversion_Probability': mean_reversion_prob,

            # Recent performance
            'Recent_Spread': recent_spread,
            'Predictions': y_pred,
            'Actual_Values': y_recent,
            'Median_Error': median_error,

            # Risk management
            'Stop_Loss_Z_score': current_z_score * 1.5,  # 50% additional deviation
            'Target_Z_score': 0,  # Mean reversion target

            # Trade setup
            'Position_Ratio': beta,
            'Z_score_Threshold': z_score_threshold
        }


async def main():
    mongodb_config = {
        "username": os.getenv('MONGO_INITDB_ROOT_USERNAME', "admin"),
        "password": os.getenv('MONGO_INITDB_ROOT_PASSWORD', "admin"),
        "host": os.getenv('MONGO_HOST', 'localhost'),
        "port": os.getenv('MONGO_PORT', 27017),
        "database": "mongodb"
    }
    candles_config = dict(connector_name='binance_perpetual',
                          interval='15m',
                          days=7,
                          batch_size=20,
                          sleep_time=5.0)
    task_config = {
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
        "db_config": mongodb_config,
        "candles_config": candles_config
    }
    task = CointegrationTask(name="cointegration_task",
                             frequency=timedelta(hours=1),
                             config=task_config)
    asyncio.run(task.execute())


if __name__ == "__main__":
    asyncio.run(main())
