import solara
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from abm_synthpopgen import PopulationModel

model = PopulationModel(
    output_csv="output.csv",
    shapefile="data_gis/output_verificati/lazio_ASL.shp",
    sep=";"
)

# Display a sample to keep rendering responsive
rng = np.random.default_rng(42)
agents = list(model.agents)
indices = rng.choice(
    len(agents),
    size=min(5000, len(agents)),
    replace=False
)

df_agents = pd.DataFrame([
    {
        "x": agents[i].pos[0],
        "y": agents[i].pos[1],
        "unit": agents[i].unit,
        "hpt": agents[i].hpt,
        "hf": agents[i].hf
    }
    for i in indices
])


@solara.component
def Page():

    selected_hpt, set_selected_hpt = solara.use_state("all")
    selected_asl, set_selected_asl = solara.use_state("all")

    solara.Title("Lazio Healthcare ABM")

    solara.Markdown(
        f"**Population:** {len(model.agents):,} agents"
    )

    with solara.Row():
        solara.Select(
            label="ASL",
            values=["all"] + sorted(df_agents["unit"].unique()),
            value=selected_asl,
            on_value=set_selected_asl
        )

        solara.Select(
            label="Hypertension",
            values=["all", "yes", "no"],
            value=selected_hpt,
            on_value=set_selected_hpt
        )

    filtered = df_agents.copy()

    if selected_asl != "all":
        filtered = filtered[filtered["unit"] == selected_asl]

    if selected_hpt != "all":
        filtered = filtered[filtered["hpt"] == selected_hpt]

    fig = go.Figure()

    # Plot ASL polygon boundaries
    for unit, polygon in model.asl_polygons.items():

        polygons = (
            list(polygon.geoms)
            if polygon.geom_type == "MultiPolygon"
            else [polygon]
        )

        for part in polygons:
            x, y = part.exterior.xy

            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(y),
                    mode="lines",
                    line=dict(color="black", width=1),
                    name=unit,
                    showlegend=False,
                    hoverinfo="name"
                )
            )

    # Plot agents
    fig.add_trace(
        go.Scattergl(
            x=filtered["x"],
            y=filtered["y"],
            mode="markers",
            marker=dict(
                size=3,
                color="royalblue",
                opacity=0.5
            ),
            text=filtered["unit"],
            hovertemplate="ASL: %{text}<extra></extra>",
            name="Agents"
        )
    )

    fig.update_layout(
        title="Synthetic population across Lazio ASLs",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1
        ),
        height=700,
        template="plotly_white"
    )

    solara.FigurePlotly(fig)