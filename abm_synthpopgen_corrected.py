"""Synthetic population ABM for Lazio ASLs using Mesa and Mesa-Geo.

Dependencies: mesa, mesa-geo, geopandas, shapely, pandas, numpy.
Place the full shapefile set (.shp, .shx, .dbf, .prj) in the GIS folder.
"""

from pathlib import Path
from collections import Counter

import mesa
import mesa_geo as mg
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


class PersonAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs, **attrs):
        super().__init__(model=model, geometry=geometry, crs=crs)
        for key, value in attrs.items():
            setattr(self, key, value)
        self.unit = str(self.unit).strip().upper()

    def step(self):
        pass


class ASLAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs, unit):
        super().__init__(model=model, geometry=geometry, crs=crs)
        self.unit = unit

    def step(self):
        pass


class PopulationModel(mesa.Model):
    def __init__(self, output_csv: str, shapefile: str, sep=";", n_target=5_000, seed=42):
        super().__init__(seed=seed)
        rng = np.random.default_rng(seed)

        # Read and optionally scale aggregated synthetic population.
        df = pd.read_csv(output_csv, sep=sep, dtype=str)
        required = {"unit", "value"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing CSV columns: {required - set(df.columns)}")
        df["value"] = pd.to_numeric(df["value"], errors="raise").astype(int)
        if (df["value"] < 0).any() or df["value"].sum() <= 0:
            raise ValueError("Population frequencies must be nonnegative with a positive total")
        df["unit"] = df["unit"].str.strip().str.upper()
        if df["unit"].isna().any():
            raise ValueError("CSV contains missing unit values")

        if n_target is not None:
            if n_target <= 0:
                raise ValueError("n_target must be positive or None")
            probabilities = df["value"].to_numpy() / df["value"].sum()
            df["value"] = rng.multinomial(n_target, probabilities)
        df = df.loc[df["value"] > 0].copy()

        # Read and prepare ASL polygons BEFORE creating any GeoAgents/GeoSpace.
        self.asl = gpd.read_file(shapefile)
        if "DENOMINAZI" not in self.asl.columns:
            raise ValueError("Shapefile is missing the DENOMINAZI field (check its .dbf file)")
        if self.asl.crs is None:
            raise ValueError("Shapefile has no CRS")
        self.asl["DENOMINAZI"] = (
            self.asl["DENOMINAZI"].astype(str).str.strip().str.upper()
        )
        if self.asl.crs.is_geographic:
            self.asl = self.asl.to_crs(self.asl.estimate_utm_crs())
        self.asl = self.asl.dissolve(by="DENOMINAZI")
        self.asl_polygons = self.asl.geometry.to_dict()

        # Validate ASL matching before allocating thousands of agents.
        missing = set(df["unit"]) - set(self.asl_polygons)
        if missing:
            raise ValueError(f"Population units absent from shapefile: {sorted(missing)}")

        self.space = mg.GeoSpace(crs=self.asl.crs)
        self.asl_agents = [
            ASLAgent(self, geometry=polygon, crs=self.asl.crs, unit=unit)
            for unit, polygon in self.asl_polygons.items()
        ]
        self.space.add_agents(self.asl_agents)

        df_agents = (
            df.loc[df.index.repeat(df["value"])]
              .drop(columns="value")
              .reset_index(drop=True)
        )
        self.char_cols = list(df_agents.columns)
        self.initialize_population(df_agents, rng)

        # Reporter groups only people, not ASL polygon agents.
        def cross_counts(model):
            counts = Counter(
                tuple(getattr(agent, col) for col in model.char_cols)
                for agent in model.person_agents
            )
            return {
                "|".join(f"{col}={value}" for col, value in zip(model.char_cols, profile)): count
                for profile, count in counts.items()
            }

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "N": lambda m: len(m.person_agents),
                "cross_counts": cross_counts,
            }
        )
        self.datacollector.collect(self)

    def initialize_population(self, df_agents, rng):
        """Create point GeoAgents within their matching ASL polygons."""
        self.person_agents = []
        bounds = {unit: polygon.bounds for unit, polygon in self.asl_polygons.items()}

        for attrs in df_agents.to_dict("records"):
            unit = attrs["unit"]
            polygon = self.asl_polygons[unit]
            minx, miny, maxx, maxy = bounds[unit]
            while True:
                point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if polygon.covers(point):
                    break
            self.person_agents.append(
                PersonAgent(self, geometry=point, crs=self.asl.crs, **attrs)
            )

        self.space.add_agents(self.person_agents)

    def step(self):
        self.agents_by_type[PersonAgent].shuffle_do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    model = PopulationModel(
        output_csv=str(base_dir / "output.csv"),
        shapefile=str(base_dir / "data_gis" / "output_verificati" / "lazio_ASL.shp"),
        sep=";",
        n_target=100_000,
        seed=42,
    )
    print("Population agents:", len(model.person_agents))
    print("ASL polygon agents:", len(model.asl_agents))
    print("Total Mesa agents:", len(model.agents))
    first = model.person_agents[0]
    print("First agent ASL:", first.unit)
    print("First agent inside its ASL:", model.asl_polygons[first.unit].covers(first.geometry))
    model_vars = model.datacollector.get_model_vars_dataframe()
    rows = [
        {"step": step, "profile": profile, "count": count}
        for step, counts in model_vars["cross_counts"].items()
        for profile, count in counts.items()
    ]
    cross_df = pd.DataFrame(rows)
