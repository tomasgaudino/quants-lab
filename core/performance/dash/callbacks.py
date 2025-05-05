import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, html, dash_table
from core.performance.dash.layout import global_layout, detail_layout


def register_callbacks(app):
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_main_content(tab):
        if tab == 'global-data-tab':
            return global_layout
        elif tab == 'explore-tab':
            return detail_layout
        else:
            return html.Div("Tab not found")

    @app.callback(
        Output('treemap-graph', 'figure'),
        Input('path-dropdown', 'value'),
        # prevent_initial_call=True
    )
    def update_treemap(selected_path):
        df_summary = pd.read_csv("assets/df_summary.csv")
        fig = px.treemap(
            df_summary,
            path=selected_path,
            values=None,
            color="controller_id"
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=800)
        return fig

    @app.callback(
        Output('detail-tabs-content', 'children'),
        [Input('detail-tabs', 'value')]
    )
    def render_explore_content(tab):
        css = [
            {'selector': '.dash-spreadsheet',
             'rule': 'background-color: #1C1917; color: #FFFFFF; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; border: none;'},
            {'selector': '.dash-header',
             'rule': 'background-color: #1C1917; color: #FFFFFF; font-weight: bold; border-bottom: 1px solid #444444;'},
            {'selector': '.dash-cell',
             'rule': 'background-color: #1C1917; color: #FFFFFF; border-bottom: 1px solid #444444; padding: 10px; text-align: center;'},
        ]
        style_cell = {
            'padding': '10px',
            'textAlign': 'center',
        }
        style_table = {'overflowX': 'auto'}

        executors_df = pd.DataFrame({'Executor': ['X', 'Y'], 'Status': ['Running', 'Stopped']})
        trades_df = pd.DataFrame({'Trade ID': [101, 102], 'Profit': [120, -50]})

        if tab == 'controllers':
            controllers_df = pd.DataFrame([
                {'controller_id': 'binance-perpetual||MELANIA-USDT||IP-USDT||2025||isoweek11_3-0140',
                 'timestamp': 1741743981.0634391,
                 'controller_type': 'generic',
                 'database_id': 'binance_perpetual-2025-11-3-0140-2025.sqlite',
                 'controller_name': 'stat_arb',
                 'global_pnl': np.random.randn(),
                 'total_volume': np.random.randint(10, 10000)},
                {'controller_id': 'binance-perpetual||WLD-USDT||TST-USDT||2025||isoweek11_7-1140',
                 'timestamp': 1742125773.6928542,
                 'controller_type': 'generic',
                 'database_id': 'binance_perpetual-2025-11-7-1140-2025.sqlite',
                 'controller_name': 'stat_arb',
                 'global_pnl': np.random.randn(),
                 'total_volume': np.random.randint(10, 10000)},
                {'controller_id': 'binance-perpetual||INJ-USDT||POPCAT-USDT||2025||isoweek11_4-1740',
                 'timestamp': 1741887635.0314603,
                 'controller_type': 'generic',
                 'database_id': 'binance_perpetual-2025-11-4-1740-2025.sqlite',
                 'controller_name': 'stat_arb',
                 'global_pnl': np.random.randn(),
                 'total_volume': np.random.randint(10, 10000)},
                {'controller_id': 'binance-perpetual||S-USDT||WAL-USDT||2025||isoweek16_2-1420',
                 'timestamp': 1744726847.7603877,
                 'controller_type': 'generic',
                 'database_id': 'binance_perpetual-2025-16-2-1420-2025.sqlite',
                 'controller_name': 'stat_arb',
                 'global_pnl': np.random.randn(),
                 'total_volume': np.random.randint(10, 10000)},
                {'controller_id': 'binance-perpetual||AUCTION-USDT||IP-USDT||2025||isoweek16_2-1420',
                 'timestamp': 1744726847.7887366,
                 'controller_type': 'generic',
                 'database_id': 'binance_perpetual-2025-16-2-1420-2025.sqlite',
                 'controller_name': 'stat_arb',
                 'global_pnl': np.random.randn(),
                 'total_volume': np.random.randint(10, 10000)}]
            )
            return dash_table.DataTable(
                id='controllers-table',
                data=controllers_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in controllers_df.columns],
                css=css,
                style_table=style_table,
                style_cell=style_cell,
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{global_pnl} < 0'},
                        'color': 'red',
                    },
                    {
                        'if': {'filter_query': '{global_pnl} >= 0'},
                        'color': 'lightgreen',
                    },
                ],
            )
        elif tab == 'executors':
            executors_df = pd.DataFrame([
                {
                    'executor_id': '8i8b5KyPZYjnCWwiMT6M9gRcbPgeQdof4r5QVePvcoA4',
                    'timestamp': 1741743984.5528204,
                    'type': 'grid_executor',
                    'close_type': 5,
                    'close_timestamp': 1741744475,
                    'status': 4,
                    'net_pnl_pct': -0.004740562892299736,
                    'net_pnl_quote': -3.47797346,
                    'cum_fees_quote': 0.33948945999999997,
                    'filled_amount_quote': 733.66255,
                    'is_active': 0,
                    'is_trading': 0,
                    'controller_id': 'binance-perpetual||MELANIA-USDT||IP-USDT||2025||isoweek11_3-0140',
                    'database_id': 'binance_perpetual-2025-11-3-0140-2025.sqlite'},
                {
                    'executor_id': '7zh6xCBHNCHp9NDonbe6X5qmErkydZLvsg5mj18ihCP3',
                    'timestamp': 1742125777.0983543,
                    'type': 'grid_executor',
                    'close_type': 5,
                    'close_timestamp': 1742127435,
                    'status': 4,
                    'net_pnl_pct': 0.0008636335888812302,
                    'net_pnl_quote': 0.6309821,
                    'cum_fees_quote': 0.3322179,
                    'filled_amount_quote': 730.6132,
                    'is_active': 0,
                    'is_trading': 0,
                    'controller_id': 'binance-perpetual||WLD-USDT||TST-USDT||2025||isoweek11_7-1140',
                    'database_id': 'binance_perpetual-2025-11-7-1140-2025.sqlite'},
                {
                    'executor_id': 'EUbheXiX8bSKszhzYrqfU7VVia2zLKKSi3H2xBqr5dvr',
                    'timestamp': 1742125777.0983543,
                    'type': 'grid_executor',
                    'close_type': 5,
                    'close_timestamp': 1742127435,
                    'status': 4,
                    'net_pnl_pct': 0.0026418244924651035,
                    'net_pnl_quote': 2.73898023,
                    'cum_fees_quote': 0.51521977,
                    'filled_amount_quote': 1036.776,
                    'is_active': 0,
                    'is_trading': 0,
                    'controller_id': 'binance-perpetual||WLD-USDT||TST-USDT||2025||isoweek11_7-1140',
                    'database_id': 'binance_perpetual-2025-11-7-1140-2025.sqlite'},
                {
                    'executor_id': 'AFjMKvjRU7zkhUW5FFFyFsy9KS6iduQ7uwDLtDpMh9Uz',
                    'timestamp': 1741887640.780641,
                    'type': 'grid_executor',
                    'close_type': 5,
                    'close_timestamp': 1741888110,
                    'status': 4,
                    'net_pnl_pct': -0.002456595360975368,
                    'net_pnl_quote': -1.77384413,
                    'cum_fees_quote': 0.34424413,
                    'filled_amount_quote': 722.0742,
                    'is_active': 0,
                    'is_trading': 0,
                    'controller_id': 'binance-perpetual||INJ-USDT||POPCAT-USDT||2025||isoweek11_4-1740',
                    'database_id': 'binance_perpetual-2025-11-4-1740-2025.sqlite'
                }
            ])
            return dash_table.DataTable(
                id='executors-table',
                data=executors_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in executors_df.columns],
                css=css,
                style_table=style_table,
                style_cell=style_cell,
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{net_pnl_pct} < 0'},
                        'color': 'red',
                    },
                    {
                        'if': {'filter_query': '{net_pnl_pct} >= 0'},
                        'color': 'lightgreen',
                    },
                ]
            )
        elif tab == 'trades':
            trades_df = pd.DataFrame({
                'Order No.': ['31934627335', '31934627336', '25703013814'],
                'Time': ['2025-04-21 20:02:47', '2025-04-21 20:02:47', '2025-04-21 19:17:44'],
                'Symbol': ['FILUSDT Perp', 'FILUSDT Perp', 'ETCUSDT Perp'],
                'Side': ['Close Short', 'Close Short', 'Close Long'],
                'Price': [2.589, 2.588, 15.609],
                'Quantity': ['163.107 USDT', '153.210 USDT', '493.245 USDT'],
                'Fee': ['0.08155350 USDT', '0.07660480 USDT', '0.24662220 USDT'],
                'Role': ['Taker', 'Taker', 'Taker'],
                'Realized Profit': ['2.92855023 USDT', '2.81110752 USDT', '-5.44291500 USDT'],
            })

            return dash_table.DataTable(
                id='trades-table',
                data=trades_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in trades_df.columns],
                css=css,
                style_table=style_table,
                style_cell=style_cell,
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Side} = "Close Long"'},
                        'color': 'red',
                    },
                    {
                        'if': {'filter_query': '{Side} = "Close Short"'},
                        'color': 'lightgreen',
                    },
                ],
            )
        else:
            return html.Div("Tab not found")
