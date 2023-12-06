# Author: Felipe Reibnitz Willemann
from model import GFN1_xTB
from pathlib import Path
import os

# defining relevant paths
src_path = Path(__file__).parent
cwd_path = Path(os.getcwd())
par_file = src_path / "parameters.dat"

# get coordinate files directory path
coord_dir = cwd_path / Path(
    input(
        "Enter relative path of directory with coordinate files (from working directory): "
    )
)

# get output directory path
out_option = input("Save outputs on current working directory? [y/n] ")
if out_option.upper() == "Y":
    out_dir = cwd_path / "outputs"
elif out_option.upper() == "N":
    out_dir = src_path.parent / "outputs"
else:
    print("Unrecognized option, aborting.")
    exit()

# check paths
if not os.path.exists(coord_dir):
    print("ERROR: No coordinates directory found!")
    exit()

if not os.path.exists(par_file):
    print("ERROR: No parameters file found!")
    exit()

if not os.path.exists(out_dir):
    print("INFO: Outputs directory not found, a new one will be made.")
    os.mkdir(out_dir)

print(f"INFO: Outputs are going to be saved on {out_dir}.")

# get max number of iterations
maxiter = int(input("Enter maximum number of SCF iterations: "))

# initializing model
model = GFN1_xTB(par_file=par_file)

# run calculations for each geometry
for file in os.listdir(coord_dir):
    base, ext = os.path.splitext(file)
    if ext != ".xyz":
        continue

    geom_file = os.path.join(coord_dir, file)
    out_file = os.path.join(out_dir, f"{base}.out")

    print(f"INFO: Loading {base} geometry.")
    load_result = model.load_geometry(geom_file=geom_file)
    if not load_result:
        print("WARNING: Calculation aborted!")
        continue

    print(f"INFO: Staring {base} SCF calculation.")
    model.scf(maxiter=maxiter, out_file=out_file)
