from dash import Dash, html, Output, Input

from core.performance.dash.layout import root_layout, global_layout, detail_layout

app = Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Requires Dash 2.17.0 or later
app.layout = html.Div(
    children=[
        html.Div(id='main-section',
                 children=root_layout)
    ],
    style={
        'padding': '50px',
        'marginTop': '20px'
    })


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


if __name__ == '__main__':
    app.run(debug=True)
