from typing import Any, Dict, List

import dash_echarts
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc


def instance_metric(legend: str, value: Any) -> html.Div:
    metric_name = html.Span(legend, className='legend')
    separator = html.Div(className='separator')
    value_str = html.Span(value, className='value')

    component_children = html.Div(children=[metric_name, separator, value_str],
                                  className='metric-inner')
    component = html.Div(children=component_children,
                         className='metric-box',
                         style={
                             'flex': '1',
                             'margin': '0 5px'
                         })
    return component


def section_metric(legend: str, value: Any) -> html.Div:
    component = html.Div(children=[
                             html.H4(legend),
                             html.P(value)
                         ],
                         className='metric-box'
                         )
    return component


def plotly_scatter(data: List[Dict[str, Any]],
                   title: str,
                   height: int = 400,
                   dark: bool = True,
                   static_plot: bool = True) -> html.Div:
    if dark:
        layout = {
            'template': 'plotly_dark',
            'title': title,
            'height': height,
            'paper_bgcolor': '#242120',
            'plot_bgcolor': '#242120',
            'font': {'color': '#ffffff'},
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Value'},
        }
    else:
        layout = {}
    sample_chart = dcc.Graph(
        config={"staticPlot": static_plot},
        figure={'data': data, 'layout': layout},
        className='dash-graph',
        style={'height': f'{height}px', 'width': '100%'}
    )

    component = html.Div(sample_chart, style={'flex': '1', 'minWidth': '0', 'height': '100%'})
    return component


import random
from dash import html
import dash_echarts

def echart_candlestick():
    # Raw data
    raw_data = [
        ['2013/1/24', 2320.26, 2320.26, 2287.3, 2362.94],
        ['2013/1/25', 2300, 2291.3, 2288.26, 2308.38],
        ['2013/1/28', 2295.35, 2346.5, 2295.35, 2346.92],
        ['2013/1/29', 2347.22, 2358.98, 2337.35, 2363.8],
        ['2013/1/30', 2360.75, 2382.48, 2347.89, 2383.76],
        ['2013/1/31', 2383.43, 2385.42, 2371.23, 2391.82],
        ['2013/2/1', 2377.41, 2419.02, 2369.57, 2421.15],
        ['2013/2/4', 2425.92, 2428.15, 2417.58, 2440.38],
        ['2013/2/5', 2411, 2433.13, 2403.3, 2437.42],
        ['2013/2/6', 2432.68, 2434.48, 2427.7, 2441.73],
        ['2013/2/7', 2430.69, 2418.53, 2394.22, 2433.89],
        ['2013/2/8', 2416.62, 2432.4, 2414.4, 2443.03],
        ['2013/2/18', 2441.91, 2421.56, 2415.43, 2444.8],
        ['2013/2/19', 2420.26, 2382.91, 2373.53, 2427.07],
        ['2013/2/20', 2383.49, 2397.18, 2370.61, 2397.94],
        ['2013/2/21', 2378.82, 2325.95, 2309.17, 2378.82],
        ['2013/2/22', 2322.94, 2314.16, 2308.76, 2330.88],
        ['2013/2/25', 2320.62, 2325.82, 2315.01, 2338.78],
        ['2013/2/26', 2313.74, 2293.34, 2289.89, 2340.71]
    ]

    category_data = [d[0] for d in raw_data]
    values = [d[1:] for d in raw_data]
    volume = [random.randint(500, 2000) for _ in raw_data]  # fake volume data

    # Random trades (mark 5 random buys and sells)
    buy_indices = random.sample(range(len(raw_data)), 5)
    sell_indices = random.sample(range(len(raw_data)), 5)
    buy_points = [[category_data[i], values[i][1]] for i in buy_indices]  # use 'close' price
    sell_points = [[category_data[i], values[i][1]] for i in sell_indices]

    option = {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"data": ["OHLC", "Volume", "Buy", "Sell"], "textStyle": {"color": "#ffffff"}},
        "grid": [
            {"left": "10%", "right": "10%", "height": "60%"},
            {"left": "10%", "right": "10%", "top": "75%", "height": "15%"}
        ],
        "xAxis": [
            {
                "type": "category",
                "data": category_data,
                "boundaryGap": False,
                "axisLine": {"onZero": False, "lineStyle": {"color": "#aaa"}},
                "splitLine": {"show": False},
                "min": "dataMin",
                "max": "dataMax",
                "gridIndex": 0
            },
            {
                "type": "category",
                "gridIndex": 1,
                "data": category_data,
                "axisLine": {"onZero": False, "lineStyle": {"color": "#aaa"}},
                "splitLine": {"show": False}
            }
        ],
        "yAxis": [
            {
                "scale": True,
                "splitArea": {"show": False},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.2)"}},
                "axisLine": {"lineStyle": {"color": "#aaa"}},
                "gridIndex": 0
            },
            {
                "scale": True,
                "gridIndex": 1,
                "splitNumber": 2,
                "axisLine": {"lineStyle": {"color": "#aaa"}},
                "splitLine": {"show": False}
            }
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1], "start": 50, "end": 100},
            {"show": True, "xAxisIndex": [0, 1], "type": "slider", "top": "90%", "start": 50, "end": 100}
        ],
        "series": [
            {
                "name": "OHLC",
                "type": "candlestick",
                "data": values,
                "itemStyle": {
                    "color": "#ec0000",
                    "color0": "#00da3c",
                    "borderColor": "#8A0000",
                    "borderColor0": "#008F28"
                },
            },
            {
                "name": "Buy",
                "type": "scatter",
                "data": buy_points,
                "symbolSize": 10,
                "itemStyle": {"color": "#00FF00"}
            },
            {
                "name": "Sell",
                "type": "scatter",
                "data": sell_points,
                "symbolSize": 10,
                "itemStyle": {"color": "#FF0000"}
            },
            {
                "name": "Volume",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": volume,
                "itemStyle": {"color": "#7777ff"}
            }
        ]
    }

    component = html.Div([
        dash_echarts.DashECharts(
            option=option,
            style={"height": "600px", "flex": 1, "width": "100%"},
        ),
    ])
    return component



