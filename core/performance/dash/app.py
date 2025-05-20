import asyncio
import os

from dash import Dash, html, dcc
from dotenv import load_dotenv

from core.performance.dash.backend import DashBackend
from core.performance.dash.callbacks import DashCallbacks
from core.performance.dash.layout import root_layout
from core.performance.sync_manager import DatabaseSyncManager

load_dotenv()


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
app.layout = html.Div(
    children=[
        dcc.Store(id='load-trigger', data=True),  # dummy callback for no-input components
        html.Div(id='main-section',
                 children=root_layout)
    ],
    style={
        'padding': '50px',
        'marginTop': '20px'
    })


root_path = os.path.abspath(os.path.join(os.getcwd(), '../../../'))
sync_manager = DatabaseSyncManager(root_path)
backend = DashBackend(root_path=root_path, sync_manager=sync_manager)
backend.sync_manager.update_all()
asyncio.run(backend.connect_backend_api_clients())
callbacks = DashCallbacks(app, backend)
callbacks.register_callbacks()


if __name__ == '__main__':
    app.run(debug=True)
