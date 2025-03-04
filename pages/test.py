from dash import (
    register_page,
    html,
)

import dash
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import math
import uuid

from dash import dcc

from consts.u1_values import u1_values

from utils.components import (
    create_example_card,
)

register_page(__name__)

layout = html.Div(
    [
        create_example_card("u122", 0, 1, 0.5, 0.01, 2, u1_values),
    ]
)