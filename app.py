import os

# Force a backend compatible with the standalone executable
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import solara
import pandas as pd

from mesa.visualization import SolaraViz, Slider
from mesa_geo.visualization import make_geospace_component

from abm_synthpopgen_corrected import (
    PopulationModel,
    PersonAgent,
    ASLAgent
)

# Folder containing app.py
BASE_DIR = Path(__file__).resolve().parent


# Define how agents appear on the map
def agent_portrayal(agent):

    if isinstance(agent, ASLAgent):
        return {
            "color": "black",
            "weight": 2,
            "fillColor": "lightblue",
            "fillOpacity": 0.15,
            "description": f"ASL: {agent.unit}"
        }

    if isinstance(agent, PersonAgent):

        if agent.hpt == "yes" and agent.hf == "yes":
            color = "purple"
        elif agent.hf == "yes":
            color = "red"
        elif agent.hpt == "yes":
            color = "orange"
        else:
            color = "blue"

        age_labels = {
            "0029": "0–29",
            "3059": "30–59",
            "60100": "60–100"
        }

        age_label = age_labels.get(str(agent.age), str(agent.age))

        return {
            "radius": 4,
            "color": color,
            "fillColor": color,
            "fillOpacity": 0.8,
            "weight": 1,
            "description": (
                f"ID: {agent.unique_id}<br>"
                f"Gender: {agent.gender}<br>"
                f"Age: {age_label}<br>"
                f"HPT: {agent.hpt}<br>"
                f"HF: {agent.hf}<br>"
                f"ASL: {agent.unit}"
            )
        }

@solara.component
def PopulationTable(model):

    # Extract only population agents (exclude ASL polygons)
    df = pd.DataFrame([
        {
            "ASL": agent.unit,
            "Gender": agent.gender,
            "Age": agent.age,
            "HPT": agent.hpt,
            "HF": agent.hf
        }
        for agent in model.agents_by_type[PersonAgent]
    ])

    # Count agents for each combination
    table = (
        df.groupby(
            ["ASL", "Gender", "Age", "HPT", "HF"],
            dropna=False
        )
        .size()
        .reset_index(name="N")
        .sort_values(
            ["ASL", "Gender", "Age", "HPT", "HF"]
        )
    )

    solara.Markdown("### Synthetic population by ASL")

    solara.DataFrame(
        table,
        items_per_page=15
    )


# Parameters displayed in the GUI
model_params = {
    "output_csv": str(BASE_DIR / "output.csv"),
    "shapefile": str(
        BASE_DIR / "data_gis" / "output_verificati" / "lazio_ASL.shp"
    ),
    "sep": ";",
    "n_target": Slider(
        "Population size",
        5000,
        1000,
        20000,
        1000
    ),
    "seed": 42
}


# Initialize the model
model = PopulationModel(
    output_csv=model_params["output_csv"],
    shapefile=model_params["shapefile"],
    sep=";",
    n_target=5000,
    seed=42
)

@solara.component
def PopulationCounter(model):
    n_people = len(model.agents_by_type[PersonAgent])

    solara.Markdown(f"### Population: {n_people:,} agents")
    
@solara.component
def HealthLegend(model):

    solara.Markdown("""
    ### Health status legend

    🔵 **Blue** — Neither HPT nor HF

    🟠 **Orange** — HPT only

    🔴 **Red** — HF only

    🟣 **Purple** — Both HPT and HF
    """)


# Create the GUI
page = SolaraViz(
    model,
    name="Lazio Healthcare ABM",
    model_params=model_params,
    components=[
        make_geospace_component(
            agent_portrayal,
            zoom=8,
            scroll_wheel_zoom=True
        ),
        PopulationCounter,
        HealthLegend,
        PopulationTable
    ]
)