from dash import (
    register_page,
    html,
)

from utils.components import (
    create_example_image_graph
)

register_page(__name__, path="/")

layout = html.Div(
    [
        create_example_image_graph()
    ]
)