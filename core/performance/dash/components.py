from typing import Any, Dict, List

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


def plotly_chart(data: List[Dict[str, Any]], title: str, height: int = 400, dark: bool = True) -> html.Div:
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
        config={"staticPlot": True},
        figure={'data': data, 'layout': layout},
        className='dash-graph',
        style={'height': '400px', 'width': '100%'}
    )

    component = html.Div(sample_chart, style={'flex': '1', 'minWidth': '0', 'height': '400px'})
    return component
