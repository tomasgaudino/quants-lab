import asyncio
import contextlib
import io
import time

from cachetools import TTLCache
import pandas as pd
import plotly.express as px
from dash import Input, Output, html, dash_table, State, Dash
from dash.exceptions import PreventUpdate

from core.performance.dash.backend import DashBackend
from core.performance.dash.layout import global_layout, detail_layout
from core.performance.sync_manager import ServerHandler


class DashCallbacks:
    cache = TTLCache(maxsize=1, ttl=300)
    table_css = [
        {'selector': '.dash-spreadsheet',
         'rule': 'background-color: #1C1917; color: #FFFFFF; font-family: Segoe UI, Tahoma, Geneva, Verdana, sans-serif; border: none;'},
        {'selector': '.dash-header',
         'rule': 'background-color: #1C1917; color: #FFFFFF; font-weight: bold; border-bottom: 1px solid #444444;'},
        {'selector': '.dash-cell',
         'rule': 'background-color: #1C1917; color: #FFFFFF; border-bottom: 1px solid #444444; padding: 10px; text-align: center;'},
    ]

    def __init__(self, app: Dash, backend: DashBackend):
        self.app = app
        self.backend = backend
        self.dummy_callback = Input('load-trigger', 'data')
        self.backend.update_performance_reports()

    def register_callbacks(self):
        self.tab_navigation_callbacks()
        self.database_sync_callbacks()
        self.selected_server_callbacks()

    def database_sync_callbacks(self):
        @self.app.callback(
            Output('server-drop-down', 'options'),
            self.dummy_callback
        )
        def get_available_servers(_):
            servers_info_df = asyncio.run(self.backend.fetch_servers_info())
            server_options = []
            for _, row in servers_info_df.iterrows():
                if row["status"] == "Connected":
                    name = "🟢 " + row["name"]
                else:
                    name = "🔴 " + row["name"]
                server_options.append(name)
            self.cache["server_options"] = server_options
            return server_options

        @self.app.callback(
            Output('db-status-content', 'children'),
            Input('server-drop-down', 'value')
        )
        def show_sync_info(value):
            if value is None:
                raise PreventUpdate

            server: ServerHandler = self.backend.sync_manager.servers[value[2:]]
            dbs_to_fetch_qty = len(server.missing_dbs_names)
            if dbs_to_fetch_qty > 0:
                msg = html.Span(f"⚠️ {dbs_to_fetch_qty} databases available. Press button to fetch.",
                                style={"marginRight": "10px"})
                button = html.Button(
                    "Fetch now",
                    id="fetch-dbs-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#00cc96",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "padding": "5px 10px",
                        "cursor": "pointer"
                    }
                )
                return html.Div([msg, button], style={"display": "flex", "alignItems": "center"})
            else:
                return "✅ All databases are up-to-date."

        @self.app.callback(
            Output("notification-box", "children"),
            Input("fetch-dbs-button", "n_clicks"),
            State("server-drop-down", "value"),
            prevent_initial_call=True
        )
        def fetch_databases(n_clicks, selected_server):
            if not selected_server:
                raise PreventUpdate

            server_key = self.get_server_key(selected_server)
            server: ServerHandler = self.backend.sync_manager.servers[server_key]

            log_output = io.StringIO()
            with contextlib.redirect_stdout(log_output):
                server.fetch_missing_dbs()

            logs = log_output.getvalue().strip().splitlines()

            # Renderizamos los logs como notificaciones individuales
            notifications = [
                html.Div(log, className="notification") for log in logs if log.strip()
            ]
            return notifications

    def tab_navigation_callbacks(self):
        @self.app.callback(
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

        @self.app.callback(
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

        @self.app.callback(
            Output('detail-tabs-content', 'children'),
            [Input('detail-tabs', 'value')]
        )
        def render_explore_content(tab):
            style_cell = {
                'padding': '10px',
                'textAlign': 'center',
            }
            style_table = {'overflowX': 'auto'}
            if tab == 'executors':
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
                    css=self.table_css,
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
                    css=self.table_css,
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

    def selected_server_callbacks(self):
        selected_server_input = Input("server-drop-down", "value")
        self.hummingbot_instances_callbacks(selected_server_input)

    def hummingbot_instances_callbacks(self, selected_server_input: Input):
        @self.app.callback(
            Output('hbot-instance-running', 'children'),
            selected_server_input,
        )
        def update_hummingbot_instances(selected_server):
            if selected_server is not None:
                server_key = self.get_server_key(selected_server)
                bots_status_resp = self.backend.backend_api_clients[server_key]["data"]["bots_status"]
                if bots_status_resp:
                    bots_status = bots_status_resp["data"]
                    n_instances = len(
                        [instance_name for instance_name, data in bots_status.items()
                         if data["status"] == "running"]
                    )
                    return n_instances
            return 0

        @self.app.callback(
            Output('hbot-instance-24h', 'children'),
            selected_server_input
        )
        def update_last_24h_instances(selected_server):
            last_24h_instances = 0
            time_window = 24 * 60 * 60
            if selected_server is not None:
                server_key = self.get_server_key(selected_server)
                performance_report = self.backend.performance_reports[server_key]
                if performance_report:
                    for db_name, index in performance_report.dbs_index.items():
                        end_time = index.get("end_time")
                        if end_time >= time.time() - time_window:
                            last_24h_instances += 1
            return last_24h_instances

        @self.app.callback(
            Output("hbot-instance-7d", 'children'),
            selected_server_input
        )
        def update_last_7d_instances(selected_server):
            last_7d_instances = 0
            time_window = 7 * 24 * 60 * 60
            if selected_server is not None:
                server_key = self.get_server_key(selected_server)
                performance_report = self.backend.performance_reports[server_key]
                if performance_report:
                    for db_name, index in performance_report.dbs_index.items():
                        end_time = index.get("end_time")
                        if end_time >= time.time() - time_window:
                            last_7d_instances += 1
            return last_7d_instances

        @self.app.callback(
            Output("hbot-instance-30d", 'children'),
            selected_server_input
        )
        def update_last_30d_instances(selected_server):
            last_30d_instances = 0
            time_window = 30 * 24 * 60 * 60
            if selected_server is not None:
                server_key = self.get_server_key(selected_server)
                performance_report = self.backend.performance_reports[server_key]
                if performance_report:
                    for db_name, index in performance_report.dbs_index.items():
                        end_time = index.get("end_time")
                        if end_time >= time.time() - time_window:
                            last_30d_instances += 1
            return last_30d_instances

        @self.app.callback(
            Output("hbot-instance-all-time", 'children'),
            selected_server_input
        )
        def update_all_time_instances(selected_server):
            all_time_instances = 0
            if selected_server is not None:
                server_key = self.get_server_key(selected_server)
                performance_report = self.backend.performance_reports[server_key]
                if performance_report:
                    all_time_instances = len(performance_report.dbs_index.keys())
            return all_time_instances

    @staticmethod
    def get_server_key(selected_server: str):
        return selected_server[2:] if selected_server.startswith("🔴") or selected_server.startswith(
                "🟢") else selected_server
