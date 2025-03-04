from dash import (
    register_page,
    html,
)

from consts.u1_values import u1_values
from consts.u2_values import u2_values

from utils.components import (
    create_example_card,
)

register_page(__name__, path="/")

layout = html.Div(
    [
        create_example_card("u1", 0, 1, 0.5, 0.01, 2, u1_values),
        create_example_card("u2", 0, 1, 0.5, 0.01, 2, u2_values),
    ]
)