import os
import json
from dotenv import load_dotenv
from dash import html, dcc
import core.performance.dash.components as components
import dash_daq as daq

load_dotenv()
servers = json.loads(os.getenv("BACKEND_API_SERVERS", '{"main": "localhost"}'))

# Root layout
root_layout = html.Div(children=[
    html.H1(children="Welcome back"),
    html.Div(children="Here is your trading overview"),
    html.Br(),
    html.Div(children=[
        html.Label("Server"),  # this adds the legend over the dropdown
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
                html.H3("Active / Archived Instances", style={'margin': '0', 'padding-right': '20px', 'whiteSpace': 'nowrap'}),
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
            components.plotly_chart(sample_data, "PnL Analysis")
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
            components.plotly_chart(sample_data, "Volume Analysis")
        ])
    ]),
])

# Detail layout
detail_layout = html.Div(className='section', children=[
    html.H5(children="WIP"),
])
