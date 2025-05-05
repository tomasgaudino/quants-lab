import pandas as pd
import plotly.express as px
from dash import Input, Output, html
from core.performance.dash.layout import global_layout, detail_layout


def register_callbacks(app):
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_content(tab):
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
