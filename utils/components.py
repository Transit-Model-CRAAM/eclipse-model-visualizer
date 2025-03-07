import dash
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import math
import uuid
import json

from dash import (
    html,
    dcc,
    Input,
    Output,
    State,
    Patch,
    callback_context,
    no_update,
    MATCH,
    ALL,
    set_props,
)
from dash_iconify import DashIconify
from pathlib import Path

from consts.tempohoras import tempoHoras

from eclipse.Eclipse.Star.Estrela import Estrela
from eclipse.Eclipse.Planet.Planeta import Planeta
from eclipse.Eclipse.Planet.Eclipse import Eclipse
from eclipse.Eclipse.Adjust.Model import Modelo
from eclipse.Eclipse.Adjust.Treatment import Tratamento

from kepler._core import solve

def create_connected_slider_and_number_input(page_name, variable_name, min_value, max_value, default_value, value_step, precision, disabled=False):
    slider_value = dcc.Slider(
        id=f"{page_name}_{variable_name}_slider_value",
        max=max_value,
        min=min_value,
        value=default_value,
        step=value_step,
        # precision=precision,
        updatemode="drag",
        disabled=disabled,
        marks={
            min_value: f"{min_value}",
            max_value: f"{max_value}"
        },
        tooltip={"always_visible": False}
    )
    numeric_value = dmc.NumberInput(
        id=f"{page_name}_{variable_name}_numeric_value",
        max=max_value,
        min=min_value,
        value=default_value,
        step=value_step,
        precision=precision,
        disabled=disabled,
    )

    # Callbacks

    @dash.callback(
        Output(f"{page_name}_{variable_name}_slider_value", "value"),
        Output(f"{page_name}_{variable_name}_numeric_value", "value"),
        Input(f"{page_name}_{variable_name}_slider_value", "value"),
        Input(f"{page_name}_{variable_name}_numeric_value", "value"),
        prevent_initial_call=True,
    )
    def connect_slider_and_numeric_values(slider, numeric):
        triggered_id = callback_context.triggered_id

        if triggered_id == f"{page_name}_{variable_name}_slider_value":
            return no_update, slider
        elif triggered_id == f"{page_name}_{variable_name}_numeric_value":
            return numeric, no_update
        
        return no_update

    return slider_value, numeric_value


def create_example_card(variable_name, min_value, max_value, default_value, value_step, precision, graph_values):
    slider_value, numeric_value = create_connected_slider_and_number_input("example", variable_name, min_value, max_value, default_value, value_step, precision)
    graph = dcc.Graph(
        id=f"example_{variable_name}_graph",
        figure={
            "data": [{'x': tempoHoras, 'y': graph_values[str(f"{default_value:.2f}")], 'type': 'line'}],
            "layout": {
                "xaxis": {
                    "range": [-2.5, 2.5],
                }
            }
        }
    )


    # Callbacks

    @dash.callback(
        Output(f"example_{variable_name}_graph", "figure"),
        Input(f"example_{variable_name}_slider_value", "value"),
        prevent_initial_call=True
    )
    def update_graph_with_values(value):
        patched_figure = Patch()

        patched_figure["data"][0]["y"] = graph_values[str(f"{value:.2f}")]

        return patched_figure


    # Layout

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            graph
                        ]
                    ),
                    dbc.Col(
                        [
                            slider_value,
                            numeric_value
                        ]
                    )
                ]
            )
        ]
    )


def generate_new_spot():
    raio_slider_value, raio_numeric_value = create_connected_slider_and_number_input("example_image", "raio", 0, 1, 0.05, 0.01, 2, True)
    intensidade_slider_value, intensidade_numeric_value = create_connected_slider_and_number_input("example_image", "intensidade", 0, 1, 0.5, 0.01, 2, True)
    latitude_slider_value, latitude_numeric_value = create_connected_slider_and_number_input("example_image", "latitude", -70, 70, -39.40, 0.01, 2, True)
    longitude_slider_value, longitude_numeric_value = create_connected_slider_and_number_input("example_image", "longitude", -90, 90, 40, 0.01, 2, True)

    delete_button = dmc.Button("Excluir Mancha", id="example_delete_spots", disabled=True)

    return dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    html.Div(["Raio"], className="variable-title"),
                                    raio_slider_value,
                                    raio_numeric_value
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.Div(["Intensidade"], className="variable-title"),
                                    intensidade_slider_value,
                                    intensidade_numeric_value
                                ]
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    html.Div(["Latitude"], className="variable-title"),
                                    latitude_slider_value,
                                    latitude_numeric_value
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.Div(["Longitude"], className="variable-title"),
                                    longitude_slider_value,
                                    longitude_numeric_value
                                ]
                            ),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    html.Div(
                        [
                            delete_button
                        ],
                        className="example-image-spot-delete-button align-center"
                    )
                ]
            )
        ],
        className="example-image-spot-container"
    )


def create_example_image_graph():
    raio= 373. #default (pixel)
    intensidadeMaxima=240 #default
    tamanhoMatriz = 856 #default
    raioSun=0.805 #raio da estrela em relacao ao raio do sol
    coeficienteHum=0.377
    coeficienteDois=0.024

    raioPlanJup = 1.138
    semiEixoUA = 0.031
    anguloInclinacao = 85.51
    periodo = 2.219  # day
    massPlaneta = 1.138 #em relacao ao R de jupiter
    ecc = 0
    anomalia = 0

    dtor=np.pi/180.

    # For 1st graph
    estrela_ = Estrela(raio,raioSun,intensidadeMaxima,coeficienteHum,coeficienteDois,tamanhoMatriz)

    img_rgb = np.array(estrela_.estrelaMatriz, dtype=np.uint8)

    # Ensure it's at least 2D
    if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3:  # RGB image
        img_gray = np.mean(img_rgb, axis=-1)  # Convert RGB to grayscale
    elif img_rgb.ndim == 2:  # Already grayscale
        img_gray = img_rgb
    else:
        raise ValueError(f"Unexpected image shape: {img_rgb.shape}")

    nk = 2*np.pi/(periodo * 24)    # em horas^(-1)
    Tp = periodo*anomalia/360. * 24. # tempo do pericentro (em horas)
    m = nk*(np.array(tempoHoras)-Tp)     # em radianos

    semiEixoRaioStar = (((1.469*(10**8))*semiEixoUA))/(raioSun * 696340)
    semiEixoPixel = semiEixoRaioStar * raio

    eccanom = solve(m,ecc)  # subrotina em anexo
    xs = semiEixoPixel*(np.cos(eccanom)-ecc)
    ys = semiEixoPixel*(math.sqrt(1-(ecc**2))*np.sin(eccanom))

    ang = anomalia*dtor-(np.pi/2)
    xp = xs*np.cos(ang)-ys*np.sin(ang)
    yp = xs*np.sin(ang)+ys*np.cos(ang)

    ie, = np.where(np.array(tempoHoras) == min(abs(np.array(tempoHoras))))

    xplaneta = xp-xp[ie[0]]
    yplaneta = yp*np.cos(anguloInclinacao*dtor)

    #### Intervalo para calculo do transito
    pp, = np.where((abs(xplaneta) < 1.2 * tamanhoMatriz/2) & (abs(yplaneta) < tamanhoMatriz/2)) #rearranja o vetor apenas com os pontos necessários para a análise da curva de luz
    xplan = xplaneta[pp] + tamanhoMatriz/2
    yplan = yplaneta[pp] + tamanhoMatriz/2

    kk=np.arange(tamanhoMatriz*tamanhoMatriz)
    raioPlanetaPixel = (raioPlanJup*69911)/(raioSun * 696340) * raio
    plan = np.zeros(tamanhoMatriz*tamanhoMatriz)+1.

    x0 = int(xplan[[len(yplan) // 2]])
    y0 = yplan[[len(yplan) // 2]]

    ii = np.where(((kk/tamanhoMatriz-y0)**2+(kk-tamanhoMatriz*np.fix(kk/tamanhoMatriz)-x0)**2 <= raioPlanetaPixel**2))
    plan[ii]=0.
    plan = plan.reshape(tamanhoMatriz, tamanhoMatriz) #posicao adicionada na matriz

    custom_colorscale = [
        (0.0, "black"),  # Lowest values → Black
        (0.6, "orange"),  # Mid-range values → Yellow
        (0.8, "yellow"),
        (1.0, "white")   # Highest values → White (optional)
    ]

    # Create figure with yellow-to-black colorscale
    fig = px.imshow(
        img_gray*plan, 
        color_continuous_scale=custom_colorscale
    )

    fig.update_layout(yaxis=dict(range=[0, 800], autorange=True))

    # For 2nd Graph:

    Nx = estrela_.getNx()
    Ny = estrela_.getNy()
    raioEstrelaPixel = estrela_.getRaioStar()

    planeta_ = Planeta(semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anomalia, estrela_.getRaioSun(), massPlaneta)

    eclipse = Eclipse(Nx, Ny, raioEstrelaPixel, estrela_, planeta_)

    eclipse.criarEclipse(anim=False, plot=False)

    # ELEMENTOS

    u1_slider_value, u1_numeric_value = create_connected_slider_and_number_input("example_image", "u1", 0, 1, 0.377, 0.001, 3)
    u2_slider_value, u2_numeric_value = create_connected_slider_and_number_input("example_image", "u2", 0, 1, 0.024, 0.001, 3)
    rsun_slider_value, rsun_numeric_value = create_connected_slider_and_number_input("example_image", "rsun", 0.001, 5, 0.805, 0.001, 3)
    angulo_inclinacao_slider_value, angulo_inclinacao_numeric_value = create_connected_slider_and_number_input("example_image", "angulo_inclinacao", 60, 100, 85.51, 0.01, 2)
    semi_eixo_slider_value, semi_eixo_numeric_value = create_connected_slider_and_number_input("example_image", "semi_eixo", 0.001, 2, 0.031, 0.001, 3)
    raio_planeta_slider_value, raio_planeta_numeric_value = create_connected_slider_and_number_input("example_image", "raio_planeta", 0.001, 5, 1.138, 0.001, 3)
    periodo_slider_value, periodo_numeric_value = create_connected_slider_and_number_input("example_image", "periodo", 0.001, 30, 2.219, 0.001, 3)


    save_card = html.Div(
        [
            dmc.Modal(
                children=[
                    html.Div("Já existe uma análise salva com este mesmo nome. Deseja sobrescrever?", className="example-image-save-modal-text"),
                    html.Div(
                        [
                            dmc.Button("Cancelar", color="gray", id="example_image_cancel_save", className="example-image-cancel-save"),
                            dmc.Button("Confirmar", id="example_image_confirm_save"),
                        ],
                        className="align-center",
                    )
                ],
                id="example_image_save_confirmation",
                className="example-image-save-confirmation",
            ),
            html.Div("Digite o nome da análise:", className="example-image-save-label"),
            dmc.TextInput(
                id="example_image_save_name",
                className="example-image-save-name",
            ),
            dmc.ActionIcon(
                DashIconify(
                    icon="ix:disk-filled",
                    width=40,
                    height=40,
                ),
                id="example_image_save_button",
                className="example-image-save-button"
            )
        ],
        className="example-image-save-card"
    )


    load_card = html.Div(
        [
            html.Div("Carregue uma análise:", className="example-image-load-label"),
            dmc.Select(
                id="example_image_load_name",
                value="",
                data=[],
                className="example-image-load-name"
            ),
            dmc.ActionIcon(
                DashIconify(
                    icon="flowbite:upload-solid",
                    width=40,
                    height=40,
                ),
                id="example_image_load_button",
                className="example-image-load-button"
            )
        ],
        className="example-image-load-card"
    )

    spots_card = html.Div(
        [
            dcc.Store(id="example_spots_store", data={}),
            html.Div(
                "Manchas",
                className="card-title",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dmc.Button("Adicionar Mancha", id="example_add_spots"),
                                    dmc.SegmentedControl(
                                        id="example_spots_segmented_control",
                                        value="",
                                        data=[]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            dbc.Row(
                html.Div(children=generate_new_spot(), id="example_spots_container")
            )
        ],
        className="example-image-spots-card card-container"
    )


    stars_search = html.Div(
        [
            dcc.Store(id="example_image_retrieved_curves", data={}),
            dcc.Store(id="example_image_plot_curve"),
            dbc.Row(
                [
                    dmc.TextInput(label="Nome da Estrela", id="example_image_star_name"),
                ]
            ),
            dbc.Row(
                [
                    dmc.Select(
                        id="example_image_cadence",
                        label="Cadência",
                        value="short",
                        data=[
                            {"label": "Long Cadence", "value": "long"},
                            {"label": "Short Cadence", "value": "short"},
                            {"label": "Fast Cadence (TESS)", "value": "fast"},
                            {"label": "Full-Frame Image (FFI)", "value": "ffi"},
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dmc.Select(
                        id="example_image_mission",
                        label="Missão",
                        value="",
                        data=[
                            {"label": "Kepler", "value": "Kepler"},
                            {"label": "K2", "value": "K2"},
                            {"label": "TESS", "value": "TESS"},
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    html.Div(
                        [
                            dmc.LoadingOverlay(
                                dmc.Button("Procurar Curvas", id="example_image_find_curves", disabled=True)
                            )
                        ],
                        className="example-image-find-curves-button align-center"
                    )
                ]
            ),
            dbc.Row(
                [
                    html.Div(
                        [
                            dbc.ListGroup(
                                id="example_image_curves_list_group",
                                className="example-image-curves-list-group"
                            )
                        ],
                        className="align-center",
                    )
                ]
            )
        ]
    )

    graph_image = dcc.Graph(
        id="example_image_graph",
        figure=fig,
    )

    graph_light_curve = dcc.Graph(
        id="example_light_curve_graph",
        figure={
            "data": [
                {'x': tempoHoras, 'y': eclipse.getCurvaLuz(), 'type': 'line', "showlegend": True, "name": "Curva Modelo"},
                {'x': [tempoHoras[x0]], 'y': [eclipse.getCurvaLuz()[x0]], "mode": "markers", "marker": dict(size=10, color="red", symbol="circle"), "showlegend": False}
            ],
            "layout": {
                "xaxis": {
                    "range": [tempoHoras[(len(tempoHoras) // 2) - int(len(yplan) // 2) - 5], tempoHoras[(len(tempoHoras) // 2) + int(len(yplan) // 2) + 5]],
                },
                "legend": {
                    "orientation": "h",
                },
                "margin": {"r": 45, "t": 35, "b": 65, "l": 45},
            }
        },
        className="example-image-graph-lightcurve"
    )

    x_axis_slider_value = dcc.Slider(
        id=f"example_image_x_axis_slider_value",
        max=len(yplan),
        min=0,
        value=len(yplan) // 2,
        step=1,
        marks=None,
        tooltip={"always_visible": False},
        className="example-image-x-axis-slider-value"
    )

    # CALLBACKS

    @dash.callback(
        Output("example_image_graph", "figure"),
        Input("example_image_u1_slider_value", "value"),
        Input("example_image_u2_slider_value", "value"),
        Input("example_image_rsun_slider_value", "value"),
        Input("example_image_angulo_inclinacao_slider_value", "value"),
        Input("example_image_semi_eixo_slider_value", "value"),
        Input("example_image_raio_planeta_slider_value", "value"),
        Input("example_image_periodo_slider_value", "value"),
        Input("example_image_x_axis_slider_value", "value"),
        Input("example_spots_store", "data"),
        prevent_initial_call=True
    )
    def update_graph_with_values(coeficienteHum, coeficienteDois, raioSun, anguloInclinacao, semiEixoUA, raioPlanJup, periodo, x0, spots_store):
        raio= 373. #default (pixel)
        intensidadeMaxima=240 #default
        tamanhoMatriz = 856 #default
        dtor=np.pi/180.

        patched_figure = Patch()

        # Criando Gráfico
        estrela_ = Estrela(raio,raioSun,intensidadeMaxima,coeficienteHum,coeficienteDois,tamanhoMatriz)

        for item in spots_store.values():
            mancha = Estrela.Mancha(item["intensidade"], item["raio"], item["latitude"], item["longitude"])
            estrela_.addMancha(mancha)

        estrela_.criaEstrelaManchada()

        img_rgb = np.array(estrela_.estrelaMatriz, dtype=np.uint8)

        # Ensure it's at least 2D
        if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3:  # RGB image
            img_gray = np.mean(img_rgb, axis=-1)  # Convert RGB to grayscale
        elif img_rgb.ndim == 2:  # Already grayscale
            img_gray = img_rgb
        else:
            raise ValueError(f"Unexpected image shape: {img_rgb.shape}")
        
        nk = 2*np.pi/(periodo * 24)    # em horas^(-1)
        Tp = periodo*anomalia/360. * 24. # tempo do pericentro (em horas)
        m = nk*(np.array(tempoHoras)-Tp)     # em radianos

        semiEixoRaioStar = (((1.469*(10**8))*semiEixoUA))/(raioSun * 696340)
        semiEixoPixel = semiEixoRaioStar * raio

        eccanom = solve(m,ecc)  # subrotina em anexo
        xs = semiEixoPixel*(np.cos(eccanom)-ecc)
        ys = semiEixoPixel*(math.sqrt(1-(ecc**2))*np.sin(eccanom))

        ang = anomalia*dtor-(np.pi/2)
        xp = xs*np.cos(ang)-ys*np.sin(ang)
        yp = xs*np.sin(ang)+ys*np.cos(ang)

        ie, = np.where(np.array(tempoHoras) == min(abs(np.array(tempoHoras))))

        xplaneta = xp-xp[ie[0]]
        yplaneta = yp*np.cos(anguloInclinacao*dtor)

        #### Intervalo para calculo do transito
        pp, = np.where((abs(xplaneta) < 1.2 * tamanhoMatriz/2) & (abs(yplaneta) < tamanhoMatriz/2)) #rearranja o vetor apenas com os pontos necessários para a análise da curva de luz
        xplan = xplaneta[pp] + tamanhoMatriz/2
        yplan = yplaneta[pp] + tamanhoMatriz/2

        kk=np.arange(tamanhoMatriz*tamanhoMatriz)
        raioPlanetaPixel = (raioPlanJup*69911)/(raioSun * 696340) * raio
        plan = np.zeros(tamanhoMatriz*tamanhoMatriz)+1.

        try:
            y0 = yplan[[len(yplan) // 2]]
        except:
            y0 = 0

        try:
            x0_ = int(xplan[x0])
        except:
            x0_ = 0

        ii = np.where(((kk/tamanhoMatriz-y0)**2+(kk-tamanhoMatriz*np.fix(kk/tamanhoMatriz)-x0_)**2 <= raioPlanetaPixel**2))
        plan[ii]=0.
        plan = plan.reshape(tamanhoMatriz, tamanhoMatriz) #posicao adicionada na matriz

        patched_figure["data"][0]["z"] = img_gray*plan

        return patched_figure
    

    @dash.callback(
        Output("example_light_curve_graph", "figure"),
        Output("example_image_x_axis_slider_value", "value"),
        Output("example_image_x_axis_slider_value", "max"),
        Input("example_image_u1_slider_value", "value"),
        Input("example_image_u2_slider_value", "value"),
        Input("example_image_rsun_slider_value", "value"),
        Input("example_image_angulo_inclinacao_slider_value", "value"),
        Input("example_image_semi_eixo_slider_value", "value"),
        Input("example_image_raio_planeta_slider_value", "value"),
        Input("example_image_periodo_slider_value", "value"),
        Input("example_image_x_axis_slider_value", "value"),
        State("example_image_x_axis_slider_value", "max"),
        Input("example_spots_store", "data"),
        prevent_initial_call=True
    )
    def update_graph_with_values(coeficienteHum, coeficienteDois, raioSun, anguloInclinacao, semiEixoUA, raioPlanJup, periodo, x0, max, spots_store):
        raio= 373. #default (pixel)
        intensidadeMaxima=240 #default
        tamanhoMatriz = 856 #default

        massPlaneta = 1.138 #em relacao ao R de jupiter
        ecc = 0
        anomalia = 0

        patched_figure = Patch()

        # Criando Gráfico
        estrela_ = Estrela(raio,raioSun,intensidadeMaxima,coeficienteHum,coeficienteDois,tamanhoMatriz)

        for item in spots_store.values():
            mancha = Estrela.Mancha(item["intensidade"], item["raio"], item["latitude"], item["longitude"])
            estrela_.addMancha(mancha)

        estrela_.criaEstrelaManchada()

        Nx = estrela_.getNx()
        Ny = estrela_.getNy()
        raioEstrelaPixel = estrela_.getRaioStar()

        planeta_ = Planeta(semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anomalia, estrela_.getRaioSun(), massPlaneta)

        eclipse = Eclipse(Nx, Ny, raioEstrelaPixel, estrela_, planeta_)

        eclipse.criarEclipse(anim=False, plot=False)

        nk = 2*np.pi/(periodo * 24)    # em horas^(-1)
        Tp = periodo*anomalia/360. * 24. # tempo do pericentro (em horas)
        m = nk*(np.array(tempoHoras)-Tp)     # em radianos

        semiEixoRaioStar = (((1.469*(10**8))*semiEixoUA))/(raioSun * 696340)
        semiEixoPixel = semiEixoRaioStar * raio

        eccanom = solve(m,ecc)  # subrotina em anexo
        xs = semiEixoPixel*(np.cos(eccanom)-ecc)
        ys = semiEixoPixel*(math.sqrt(1-(ecc**2))*np.sin(eccanom))

        ang = anomalia*dtor-(np.pi/2)
        xp = xs*np.cos(ang)-ys*np.sin(ang)
        yp = xs*np.sin(ang)+ys*np.cos(ang)

        ie, = np.where(np.array(tempoHoras) == min(abs(np.array(tempoHoras))))

        xplaneta = xp-xp[ie[0]]
        yplaneta = yp*np.cos(anguloInclinacao*dtor)

        #### Intervalo para calculo do transito
        pp, = np.where((abs(xplaneta) < 1.2 * tamanhoMatriz/2) & (abs(yplaneta) < tamanhoMatriz/2)) #rearranja o vetor apenas com os pontos necessários para a análise da curva de luz
        yplan = yplaneta[pp] + tamanhoMatriz/2

        patched_figure["data"][0]["y"] = eclipse.getCurvaLuz()
        try:
            x_value = (len(tempoHoras) // 2) + (x0 - int(max // 2))

            patched_figure["data"][1]["x"] = [tempoHoras[x_value]]
            patched_figure["data"][1]["y"] = [eclipse.getCurvaLuz()[x_value]]
        except:
            try:
                x_value = (len(tempoHoras) // 2) + int(len(yplan) // 2)

                patched_figure["data"][1]["x"] = [tempoHoras[x_value]]
                patched_figure["data"][1]["y"] = [eclipse.getCurvaLuz()[x_value]]
            except:
                patched_figure["data"][1]["x"] = [tempoHoras[-1]]
                patched_figure["data"][1]["y"] = [eclipse.getCurvaLuz()[-1]]


        try:
            patched_figure["layout"]["xaxis"]["range"] = [
                tempoHoras[(len(tempoHoras) // 2) - int(len(yplan) // 2) - 5],
                tempoHoras[(len(tempoHoras) // 2) + int(len(yplan) // 2) + 5]
            ]
        except:
            patched_figure["layout"]["xaxis"]["range"] = [tempoHoras[0], tempoHoras[-1]]

        try:
            x0_return = x0
        except:
            x0_return = int(len(yplan) // 2)

        if callback_context.triggered_id != "example_image_x_axis_slider_value":
            return patched_figure, x0_return, int(len(yplan))-1
        
        return patched_figure, no_update, no_update
    

    @dash.callback(
        Output("example_image_raio_slider_value", "value", allow_duplicate=True),
        Output("example_image_intensidade_slider_value", "value", allow_duplicate=True),
        Output("example_image_latitude_slider_value", "value", allow_duplicate=True),
        Output("example_image_longitude_slider_value", "value", allow_duplicate=True),
        Output("example_image_raio_slider_value", "disabled"),
        Output("example_image_intensidade_slider_value", "disabled"),
        Output("example_image_latitude_slider_value", "disabled"),
        Output("example_image_longitude_slider_value", "disabled"),
        Output("example_image_raio_numeric_value", "disabled"),
        Output("example_image_intensidade_numeric_value", "disabled"),
        Output("example_image_latitude_numeric_value", "disabled"),
        Output("example_image_longitude_numeric_value", "disabled"),
        Output("example_delete_spots", "disabled"),
        Output("example_spots_segmented_control", "value"),
        Output("example_spots_segmented_control", "data"),
        Output("example_spots_store", "data"),
        Input("example_add_spots", "n_clicks"),
        Input("example_delete_spots", "n_clicks"),
        State("example_spots_segmented_control", "data"),
        State("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def example_add_new_spot_callback(nc1, nc2, segmented_control_data, segmented_control_value, spots_store):
        triggered_id = callback_context.triggered_id

        if triggered_id == "example_add_spots":
            new_id = str(uuid.uuid4())
            if segmented_control_data == []:
                new_spot_number = 1
            else:
                new_spot_number = int(segmented_control_data[-1]["label"].split(" ")[1])+1
            
            segmented_control_data.append({"value": new_id, "label": f"Mancha {new_spot_number}"})

            spots_store[new_id] = {
                "raio": 0.05,
                "intensidade": 0.5,
                "latitude": -39.40,
                "longitude": 40.00,
            }

            return (
                spots_store[new_id]["raio"],
                spots_store[new_id]["intensidade"],
                spots_store[new_id]["latitude"],
                spots_store[new_id]["longitude"],
                *[False]*9,
                new_id,
                segmented_control_data,
                spots_store
            )
        elif triggered_id == "example_delete_spots":
            idx = next((i for i, d in enumerate(segmented_control_data) if d["value"] == segmented_control_value), None)

            del segmented_control_data[idx]
            del spots_store[segmented_control_value]

            if segmented_control_data == []:
                return (
                    *[no_update]*4,
                    *[True]*9,
                    "",
                    segmented_control_data,
                    spots_store
                )
            elif idx < len(segmented_control_data):
                new_id = segmented_control_data[idx]["value"]
            else:
                new_id = segmented_control_data[idx-1]["value"]

            return (
                spots_store[new_id]["raio"],
                spots_store[new_id]["intensidade"],
                spots_store[new_id]["latitude"],
                spots_store[new_id]["longitude"],
                *[False]*9,
                new_id,
                segmented_control_data,
                spots_store
            )

        return no_update
    

    @dash.callback(
        Output("example_spots_store", "data", allow_duplicate=True),
        Input("example_image_raio_slider_value", "value"),
        State("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def spots_raio_update_value(value, spot_id, spots_store):
        spots_store[spot_id]["raio"] = value

        return spots_store
    

    @dash.callback(
        Output("example_spots_store", "data", allow_duplicate=True),
        Input("example_image_intensidade_slider_value", "value"),
        State("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def spots_intensidade_update_value(value, spot_id, spots_store):
        spots_store[spot_id]["intensidade"] = value

        return spots_store
    

    @dash.callback(
        Output("example_spots_store", "data", allow_duplicate=True),
        Input("example_image_latitude_slider_value", "value"),
        State("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def spots_latitude_update_value(value, spot_id, spots_store):
        spots_store[spot_id]["latitude"] = value

        return spots_store
    

    @dash.callback(
        Output("example_spots_store", "data", allow_duplicate=True),
        Input("example_image_longitude_slider_value", "value"),
        State("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def spots_longitude_update_value(value, spot_id, spots_store):
        spots_store[spot_id]["longitude"] = value

        return spots_store


    @dash.callback(
        Output("example_image_raio_slider_value", "value", allow_duplicate=True),
        Output("example_image_intensidade_slider_value", "value", allow_duplicate=True),
        Output("example_image_latitude_slider_value", "value", allow_duplicate=True),
        Output("example_image_longitude_slider_value", "value", allow_duplicate=True),
        Input("example_spots_segmented_control", "value"),
        State("example_spots_store", "data"),
        prevent_initial_call=True,
    )
    def load_spots_values(spot_id, spots_store):
        if spot_id != "" and spot_id is not None:
            return (
                spots_store[spot_id]["raio"],
                spots_store[spot_id]["intensidade"],
                spots_store[spot_id]["latitude"],
                spots_store[spot_id]["longitude"]
            )
    
        return no_update

    @dash.callback(
        Output("example_image_find_curves", "disabled"),
        Input("example_image_star_name", "value"),
        Input("example_image_cadence", "value"),
        Input("example_image_mission", "value"),
        prevent_initial_call=True,
    )
    def find_curves_button_enabler(star_name, cadence, mission):
        if (
            star_name is not None
            and cadence is not None
            and mission is not None
            and star_name != ""
            and cadence != ""
            and mission != ""
        ):
            return False

        return True


    @dash.callback(
        Output("example_image_retrieved_curves", "data"),
        Output("example_image_find_curves", "style"),
        Output("example_light_curve_graph", "figure", allow_duplicate=True),
        Input("example_image_find_curves", "n_clicks"),
        State("example_image_u1_slider_value", "value"),
        State("example_image_u2_slider_value", "value"),
        State("example_image_rsun_slider_value", "value"),
        State("example_image_star_name", "value"),
        State("example_image_cadence", "value"),
        State("example_image_mission", "value"),
        State("example_light_curve_graph", "figure"),
        prevent_initial_call=True,
    )
    def load_star_data(nc1, u1, u2, rsun, star_name, cadence, mission, figure):
        raio= 373. #default (pixel)
        intensidadeMaxima=240 #default
        tamanhoMatriz = 856 #default

        estrela_ = Estrela(raio, rsun, intensidadeMaxima, u1, u2, tamanhoMatriz)

        Nx = estrela_.getNx()
        Ny = estrela_.getNy()
        raioEstrelaPixel = estrela_.getRaioStar()

        eclipse = Eclipse(Nx, Ny, raioEstrelaPixel, estrela_, planeta_)
        
        estrela_.setStarName(star_name)
        estrela_.setCadence(cadence)
        eclipse.setTempoHoras(1.)
        # lc0 = np.array(eclipse.getCurvaLuz())
        # ts0 = np.array(eclipse.getTempoHoras())

        modelo = Modelo(estrela_, eclipse, mission)

        ls_model, ts_model = modelo.eclipse_model()

        time, flux, flux_err = modelo.rd_data(0, 0)

        modelo.setTime(time)
        modelo.setFlux(flux)
        modelo.setFluxErr(flux_err)

        x0, nt = modelo.det_x0(0)

        tratamento = Tratamento(modelo)

        dur, tim, lcurve, f_err = tratamento.cut_transit_single()

        # t_p = tim[100]
        # s_lc = lcurve[100]

        # bb = np.where((t_p >= min(ts0)) & (t_p <= max(ts0)))
        # bb = np.where((ts0 >= -5.) & (ts0 <= 5.))
        # dd = np.where((t_p >= -5.) & (t_p <= 5.))

        transit_data = {}
        for i in range(int(nt)):
            try:
                selectedTransit = i #transito selecionado
                time_phased, smoothed_LC = tratamento.select_transit_smooth(selectedTransit)

                transit_data[f"Transito {i}"] = {
                    "time_phased": time_phased,
                    "smoothed_LC": smoothed_LC
                }
            except:
                pass

        patched_figure = Patch()

        for i in range(len(figure["data"])-1, 1, -1):
            del patched_figure["data"][i]

        return transit_data, no_update, patched_figure
    

    @dash.callback(
        Output("example_image_curves_list_group", "children"),
        Input("example_image_retrieved_curves", "data"),
        prevent_initial_call=True
    )
    def loading_curves_into_element(data):
        children_list = []

        for item in list(data.keys()):
            children_list.append(
                dbc.ListGroupItem(
                    item,
                    id={"type": "example_image_retrieved_curve", "id": item},
                    active=False,
                    n_clicks=0,
                    className="example-image-retrieved-curve-item"
                )
            )

        return children_list
    

    @dash.callback(
        Output({"type": "example_image_retrieved_curve", "id": MATCH}, "active"),
        Input({"type": "example_image_retrieved_curve", "id": MATCH}, "n_clicks"),
        Input({"type": "example_image_retrieved_curve", "id": MATCH}, "active"),
        State({"type": "example_image_retrieved_curve", "id": MATCH}, "children"),
        prevent_initial_call=True,
    )
    def select_curve(nc1, active, name):
        set_props("example_image_plot_curve", {"data": {"active": active, "name": name}})
        return not active
    

    @dash.callback(
        Output("example_light_curve_graph", "figure", allow_duplicate=True),
        Input("example_image_plot_curve", "data"),
        State("example_light_curve_graph", "figure"),
        State("example_image_retrieved_curves", "data"),
        prevent_initial_call=True,
    )
    def plot_curve(data, figure, curves_data):
        name = data["name"]
        active = data["active"]

        patched_figure = Patch()

        if not active:
            patched_figure["data"].append(
                {
                    "x": curves_data[name]["time_phased"],
                    "y": curves_data[name]["smoothed_LC"],
                    "type": "line",
                    "name": name,
                    "showlegend": True,
                }
            )
        else:
            for i, curve in enumerate(figure["data"]):
                try:
                    if curve["name"] == name:
                        del patched_figure["data"][i]
                        break
                except:
                    pass

        return patched_figure

    @dash.callback(
        Output("example_image_save_confirmation", "opened", allow_duplicate=True),
        Input("example_image_cancel_save", "n_clicks"),
        prevent_initial_call=True,
    )
    def example_image_close_modal(nc1):
        return False


    @dash.callback(
        Output("example_image_load_name", "data"),
        Output("example_image_save_confirmation", "opened"),
        Input("example_image_save_button", "n_clicks"),
        Input("example_image_confirm_save", "n_clicks"),
        State("example_image_save_name", "value"),
        State("example_image_load_name", "data"),
        State("example_image_u1_slider_value", "value"),
        State("example_image_u2_slider_value", "value"),
        State("example_image_rsun_slider_value", "value"),
        State("example_image_angulo_inclinacao_slider_value", "value"),
        State("example_image_semi_eixo_slider_value", "value"),
        State("example_image_raio_planeta_slider_value", "value"),
        State("example_image_periodo_slider_value", "value"),
        State("example_spots_store", "data"),
        State("example_image_star_name", "value"),
        State("example_image_cadence", "value"),
        State("example_image_mission", "value"),
    )
    def example_image_save_data(
        nc1,
        nc2,
        filename,
        load_data,
        u1,
        u2,
        rsun,
        angulo_inclinacao,
        semi_eixo,
        raio_planeta,
        periodo,
        spots,
        star_name,
        cadence,
        mission,
    ):
        triggered_id = callback_context.triggered_id
        folder_path = Path.cwd() / "analises"

        if triggered_id is None:
            if folder_path.exists() and folder_path.is_dir():
                json_files = [
                    {"label": file.stem, "value": str(file)} for file in folder_path.glob("*.json")
                ]
                return json_files, no_update
            
            return no_update
        
        elif triggered_id == "example_image_save_button":
            file_path = folder_path / f"{filename}.json"

            if file_path.exists():
                return no_update, True

            file_data = {
                "estrela": {
                    "u1": u1,
                    "u2": u2,
                    "rsun": rsun,
                    "star_name": star_name,
                    "cadence": cadence,
                    "mission": mission
                },
                "manchas": spots,
                "planeta": {
                    "angulo_inclinacao": angulo_inclinacao,
                    "semi_eixo": semi_eixo,
                    "raio_planeta": raio_planeta,
                    "periodo": periodo,
                },
            }

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(file_data, f, indent=4, ensure_ascii=False)

            load_data.append({"label": filename, "value": str(file_path)})

            return load_data, no_update

        file_path = folder_path / f"{filename}.json"

        file_data = {
            "estrela": {
                "u1": u1,
                "u2": u2,
                "rsun": rsun,
                "star_name": star_name,
                "cadence": cadence,
                "mission": mission
            },
            "manchas": spots,
            "planeta": {
                "angulo_inclinacao": angulo_inclinacao,
                "semi_eixo": semi_eixo,
                "raio_planeta": raio_planeta,
                "periodo": periodo,
            },
        }

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4, ensure_ascii=False)

        return load_data, False
    

    @dash.callback(
        Output("example_image_u1_slider_value", "value", allow_duplicate=True),
        Output("example_image_u2_slider_value", "value", allow_duplicate=True),
        Output("example_image_rsun_slider_value", "value", allow_duplicate=True),
        Output("example_image_star_name", "value", allow_duplicate=True),
        Output("example_image_cadence", "value", allow_duplicate=True),
        Output("example_image_mission", "value", allow_duplicate=True),
        Output("example_image_angulo_inclinacao_slider_value", "value", allow_duplicate=True),
        Output("example_image_semi_eixo_slider_value", "value", allow_duplicate=True),
        Output("example_image_raio_planeta_slider_value", "value", allow_duplicate=True),
        Output("example_image_periodo_slider_value", "value", allow_duplicate=True),
        Output("example_spots_store", "data", allow_duplicate=True),
        Output("example_spots_segmented_control", "data", allow_duplicate=True),
        Output("example_spots_segmented_control", "value", allow_duplicate=True),
        Output("example_image_raio_slider_value", "disabled", allow_duplicate=True),
        Output("example_image_intensidade_slider_value", "disabled", allow_duplicate=True),
        Output("example_image_latitude_slider_value", "disabled", allow_duplicate=True),
        Output("example_image_longitude_slider_value", "disabled", allow_duplicate=True),
        Input("example_image_load_button", "n_clicks"),
        State("example_image_load_name", "value"),
        prevent_initial_call=True,
    )
    def example_image_load_data(
        nc1,
        file_path_str,
    ):
        file_path = Path(file_path_str)

        with file_path.open("r", encoding="utf-8") as f:
            file_data = json.load(f)

        segmented_control_data = [
            {"value": key, "label": f"Mancha {i+1}"} 
            for i, key in enumerate(file_data["manchas"])
        ]

        if segmented_control_data:
            segmented_control_value = segmented_control_data[0]["value"]
        else:
            segmented_control_value = ""

        if segmented_control_data:
            disabled = False
        else:
            disabled = True

        return (
            file_data["estrela"]["u1"],
            file_data["estrela"]["u2"],
            file_data["estrela"]["rsun"],
            file_data["estrela"]["star_name"],
            file_data["estrela"]["cadence"],
            file_data["estrela"]["mission"],
            file_data["planeta"]["angulo_inclinacao"],
            file_data["planeta"]["semi_eixo"],
            file_data["planeta"]["raio_planeta"],
            file_data["planeta"]["periodo"],
            file_data["manchas"],
            segmented_control_data,
            segmented_control_value,
            *[disabled]*4
        )
        

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    graph_image
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.Div(
                                        x_axis_slider_value,
                                        className="example-image-x-axis-slider-container align-center"
                                    )
                                ]
                            ),
                            dbc.Row(
                                [
                                    html.Div(
                                        graph_light_curve,
                                        className="example-image-graph-lightcurve-container align-center"
                                    )
                                ]
                            ),
                        ],
                        width=4
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                load_card,
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Variáveis",
                                                        className="card-title",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Coef. de Escurecimento de Limbo 1"], className="variable-title"),
                                                                            u1_slider_value,
                                                                            u1_numeric_value
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Coef. de Escurecimento de Limbo 2"], className="variable-title"),
                                                                            u2_slider_value,
                                                                            u2_numeric_value
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Raio da estrela (relação ao raio do Sol)"], className="variable-title"),
                                                                            rsun_slider_value,
                                                                            rsun_numeric_value
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                            dbc.Col(
                                                                [
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Ângulo de Inclinação"], className="variable-title"),
                                                                            angulo_inclinacao_slider_value,
                                                                            angulo_inclinacao_numeric_value
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Semi-Eixo (Em UA)"], className="variable-title"),
                                                                            semi_eixo_slider_value,
                                                                            semi_eixo_numeric_value
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Raio do Planeta (Em Raios de Júpiter)"], className="variable-title"),
                                                                            raio_planeta_slider_value,
                                                                            raio_planeta_numeric_value
                                                                        ]
                                                                    ),
                                                                    dbc.Row(
                                                                        [
                                                                            html.Div(["Período (Em dias)"], className="variable-title"),
                                                                            periodo_slider_value,
                                                                            periodo_numeric_value
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ]
                                                    )
                                                ],
                                                className="card-container variables-container"
                                            )
                                        ]
                                    )
                                ]
                            ),
                            dbc.Row(
                                spots_card,
                                className="example-image-spots-card-remove-margin"
                            ),
                            dbc.Row(
                                save_card,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        stars_search,
                        width=2
                    )
                ]
            )
        ],
        className="example-image-container"
    )