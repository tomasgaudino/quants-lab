from dash import Dash, html

from core.performance.dash.callbacks import register_callbacks
from core.performance.dash.layout import root_layout


app = Dash(__name__, suppress_callback_exceptions=True)

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

register_callbacks(app)


if __name__ == '__main__':
    app.run(debug=True)
