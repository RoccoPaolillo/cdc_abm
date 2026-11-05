import mesa
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# class agents initialization
# **attrs dictionary from df_agents key, value
class PersonAgent(mesa.Agent):
    def __init__(self, model, **attrs):
        super().__init__(model)
        for k, v in attrs.items():
            setattr(self, k, v)
            
        if hasattr(self,"unit"):
            self.unit = str(self.unit).strip().upper()
        
        self.geometry = None
        self.pos = None

# for future schedule activation (not activated)
    def step(self):
        pass

# model initialization
class PopulationModel(mesa.Model):
    def __init__(self, output_csv: str,shapefile, sep=";"):
        super().__init__()

# read pdf and consider all variables as string except "value" integer
        df = pd.read_csv(output_csv, sep=sep, dtype=str)
        df["value"] = df["value"].astype(int)
        
        N_target = 100_000

        rng = np.random.default_rng(42)

        probabilities = df["value"].to_numpy() / df["value"].sum()

        df["value"] = rng.multinomial(
            N_target,
            probabilities
            )

        df = df[df["value"] > 0].copy()
        
        # Read ASL shapefile
        self.asl = gpd.read_file(shapefile)
        
        # Normalize names
        self.asl["DENOMINAZI"] = (
            self.asl["DENOMINAZI"]
            .astype(str)
            .str.strip()
            .str.upper()
            )
        
            # Use projected coordinates in meters
        if self.asl.crs is None:
            raise ValueError("Shapefile has no CRS")
                
        if self.asl.crs.is_geographic:
            self.asl = self.asl.to_crs(
                self.asl.estimate_utm_crs()
                )
                
        # Merge polygons for repeated ASL names
        self.asl = self.asl.dissolve(
            by = "DENOMINAZI"
            )
        
        # ASL polygon lookup
        self.asl_polygons = self.asl.geometry.to_dict()
        
        # Expand synthetic population
        df_agents = (
            df.loc[df.index.repeat(df["value"])]
            .drop(columns=["value"])
            .reset_index(drop=True)
            )
        
        self.char_cols = list(df_agents.columns)
        
        # Check ALS correspondence
        units = set(df_agents["unit"].str.strip().str.upper())
        missing = units - set(self.asl_polygons)
        
        if missing:
            raise ValueError(
                f"ASL units not found in shapefile: {missing}"
                )
            
        # Create Mesa agents
        PersonAgent.from_dataframe(self, df_agents)
        
        # Spatially locate agents
        self.initialize_positions()
                
    def initialize_positions(self):
        
        rng = np.random.default_rng(42)
        
        for agent in self.agents:
            
            polygon = self.asl_polygons[agent.unit]
            
            # Random point inside ASL
            minx, miny, maxx, maxy = polygon.bounds
            
            while True:
                x = rng.uniform(minx,maxx)
                y = rng.uniform(miny,maxy)
                
                point = Point(x, y)
                
                if polygon.covers(point):
                    break
            
            agent.geometry = point
            agent.pos = (x,y)

        # ---- Reporter: counts for each joint combination of all characteristics ----
        def cross_counts(model):
            # Convert agents -> rows 
            rows = [
                {c: getattr(a, c) for c in model.char_cols}
                for a in model.agents
            ]
            
            df = pd.DataFrame(rows)
            
            group_cols = model.char_cols
            
            g = df.groupby(group_cols, dropna=False).size()
            
            # Return as dict with readable keys
            # key like "age=3060|hpt=yes|hf=no|gender=male"
            out = {
                "|".join(f"{col}={val}" for col, val in zip(group_cols, idx)): int(cnt)
                for idx, cnt in g.items()
            }
            return out

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "N": lambda m: len(m.agents),
                "cross_counts": cross_counts
            }
        )

        # collect at t=0
        self.datacollector.collect(self)

    def step(self):
        # actual scheduler of agents' actions (inactive)
        self.agents.shuffle_do("step")
        # to collect output from the model
        self.datacollector.collect(self)


### To run the actual model
# model initialized with output.csv (df_agents in the code)    
# NOTE! The script is already set to use "output.csv" as external file generated from the SPG service
# You can set to another file
model = PopulationModel(
    output_csv = "output.csv", 
    shapefile= "data_gis/output_verificati/lazio_ASL.shp",
    sep=";")
# model.run_for(2) # to execute 3 runs (0,1,2)  
# to collect report     
model_vars = model.datacollector.get_model_vars_dataframe()

# to extract a tidy table for visualization
rows = []
for step, d in model_vars["cross_counts"].items():
    for profile, count in d.items():
        rows.append({"step": step, "profile": profile, "count": count})

cross_df = pd.DataFrame(rows)
