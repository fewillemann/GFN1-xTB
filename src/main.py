from model import GFN1_xTB
from pathlib import Path
import os

# defining relevant paths
src_path = Path(__file__).parent
par_file = f"{src_path}/parameters.dat"
coord_dir = f"{src_path.parent}/coords"
out_dir = f"{src_path.parent}/outputs"

if not os.path.exists(par_file):
    print("ERROR: No parameters file found!")
    exit()

if not os.path.exists(coord_dir):
    print("ERROR: No coordinates directory found!")
    exit()

if not os.path.exists(out_dir):
    print("WARNING: No outputs directory found, a new one will be made.")
    os.mkdir(out_dir)

# initializing model
model = GFN1_xTB(par_file=par_file)

# get max number of iterations
maxiter = int(input("Enter maximum number of SCF iterations: "))

# run calculations for each geometry
for coord in os.listdir(coord_dir):
    name = coord.split(sep=".")[0]
    geom_file = os.path.join(coord_dir, coord)
    out_file = os.path.join(out_dir, f"{name}.out")

    print(f"INFO: Loading {name} geometry.")
    load_result = model.load_geometry(geom_file=geom_file)
    if not load_result:
        continue

    print(f"INFO: Staring {name} SCF calculation.")
    model.scf(maxiter=maxiter, out_file=out_file)
