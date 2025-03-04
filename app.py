from dash import Dash, html, page_container
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div([page_container])

if __name__ == "__main__":
    app.run(debug=True)