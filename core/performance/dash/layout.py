import os
import json

import pandas as pd
from dotenv import load_dotenv
from dash import html, dcc
import core.performance.dash.components as components
from core.performance.dash.components import echart_candlestick

load_dotenv()
servers = json.loads(os.getenv("BACKEND_API_SERVERS", '{"main": "localhost"}'))

# Root layout
root_layout = html.Div(children=[
    html.Div(
        style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'},
        children=[
            html.H1("Welcome back", style={'textAlign': 'left'}),
            html.A(
                html.Img(src="assets/hummingbot_logo.png", style={'height': '160px'}),
                href="https://hummingbot.org/",
                target="_blank"  # open in new tab
            )
        ]
    ),
    html.Div(children="Here is your trading overview"),
    html.Br(),
    html.Div(children=[
        html.Label("Server"),
        dcc.Dropdown(
            options=[{'label': s, 'value': s} for s in servers.keys()],
            multi=False,
            className='dash-dropdown'
        ),
    ]),
    html.Br(),
    dcc.Tabs(
        id="main-tabs",
        value='global-data-tab',
        className='custom-tab',
        children=[
            dcc.Tab(label='Global Data', value='global-data-tab'),
            dcc.Tab(label='Explore', value='explore-tab'),
        ]
    ),
    html.Div(id="tab-content"),
])

# Sample chart
sample_data = [{'x': [1, 2], 'y': [3, 4]}]

# Global layout
global_layout = html.Div(style={"padding": "15px"}, children=[
    html.Div(className='section', children=[
        html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'width': '100%'},
            children=[
                html.H3("Hummingbot Instances", style={'margin': '0', 'padding-right': '20px', 'whiteSpace': 'nowrap'}),
                html.Div(
                    style={'display': 'flex', 'flex': '1'},
                    children=[
                        components.instance_metric("⏳ Running", 0),
                        components.instance_metric("📥 Last 24h", 0),
                        components.instance_metric("📥 Last 7d", 0),
                        components.instance_metric("📥 Last 30d", 0),
                        components.instance_metric("📥 All Time", 0),
                    ]
                ),
            ]
        ),
    ], style={'margin-bottom': '30px'}),
    # === PnL Section ===
    html.Div(className='section', children=[
        html.H3("PnL Analysis", style={"text-align": "center"}),
        html.Div(style={'display': 'flex'}, children=[
            html.Div(children=[
                components.section_metric("Total", f"$ {0.0:.2f}"),
                components.section_metric("Last 24h", f"$ {0.0:.2f}"),
                components.section_metric("Last 7d", f"$ {0.0:.2f}"),
                components.section_metric("Last 30d", f"$ {0.0:.2f}"),
            ], style={'width': '200px'}),
            components.plotly_scatter(sample_data, "PnL Analysis")
        ])
    ]),

    # === Volume Section ===
    html.Div(className='section', children=[
        html.H3("Volume Analysis", style={"text-align": "center"}),
        html.Div(style={'display': 'flex'}, children=[
            html.Div(children=[
                components.section_metric("Total", f"$ {0.0:.2f}"),
                components.section_metric("Last 24h", f"$ {0.0:.2f}"),
                components.section_metric("Last 7d", f"$ {0.0:.2f}"),
                components.section_metric("Last 30d", f"$ {0.0:.2f}"),
            ], style={'width': '200px'}),
            components.plotly_scatter(sample_data, "Volume Analysis")
        ])
    ]),
])

df_summary = pd.read_csv("assets/df_summary.csv")
initial_path = ["controller_name", "connector_name", "trading_pair", "database_id", "controller_id"]

detail_layout = html.Div(children=[
    html.H2("🎯 Navigate your own path"),
    html.Br(),
    html.Div(className="section", children=[
        html.Label("Treemap hierarchy"),
        dcc.Dropdown(
            id='path-dropdown',
            options=[{'label': col, 'value': col} for col in df_summary.columns],
            value=initial_path,
            multi=True
        ),
    ]),
    html.Div(style={'display': 'flex', 'width': '100%'}, className="section", children=[
        html.Div(
            children=[
                html.Div(components.section_metric("PnL", f"$ {0.0:.2f}"), style={'flex': '1'}),
                html.Div(components.section_metric("Volume", f"$ {0.0:.2f}"), style={'flex': '1'}),
                html.Div(components.section_metric("Total Trades", f"{334}"), style={'flex': '1'}),
                html.Div(components.section_metric("Max Draw Down", f"{32.0:.2f}%"), style={'flex': '1'}),
                html.Div(components.section_metric("Sharpe Ratio", f"{1.03:.2f}"), style={'flex': '1'}),
                html.Div(components.section_metric("Total Duration", f"17d 4h 30m"), style={'flex': '1'}),
                html.Div(components.section_metric("Date Range", f"2025-04-03 -> 2025-04-20"), style={'flex': '1'}),
            ],
            style={
                'width': '100%',
                'display': 'flex',
                'justifyContent': 'space-between',
                'gap': '10px'  # Optional: controls spacing between items
            }
        )
    ]),

    html.Div(
        style={"display": "flex", "flex": 1, "width": "100%"},
        children=[
            html.Div(
                className='section',
                style={"width": "50%", "margin": "5px", "display": "flex", "flexDirection": "column"},
                children=[
                    html.H5(children="Trading Universe"),
                    dcc.Loading(
                        id="loading-graph",
                        type="default",  # options: 'default', 'circle', 'dot', 'cube'
                        children=dcc.Graph(id='treemap-graph')
                    )
                ]),
            html.Div(
                style={"width": "50%",},
                children=[
                    html.Div(
                        className='section',
                        style={"flex": 1, "margin": "5px", "display": "flex", "flexDirection": "column"},
                        children=[
                            html.H5(children="Global PnL"),
                            components.plotly_scatter(sample_data, "Global PnL", height=400),
                        ]),
                    html.Div(
                        className='section',
                        style={"flex": 1, "margin": "5px", "display": "flex", "flexDirection": "column"},
                        children=[
                            html.H5(children="Global Volume"),
                            components.plotly_scatter(sample_data, "Total Volume", height=400),
                        ]),
                ],
            ),
        ]
    ),

    html.Div(
        className="section",
        children=[components.controllers_table()]
    ),
    html.Div(
        className="section",
        children=echart_candlestick()
    ),
    html.Div(
        className="section",
        children=[
            dcc.Tabs(className='custom-tab', id='detail-tabs', value='executors', children=[
                dcc.Tab(label='Executors', value='executors'),
                dcc.Tab(label='Trades', value='trades'),
            ]),
            html.Div(
                id="detail-tabs-content",
            )
        ]
    ),
])

