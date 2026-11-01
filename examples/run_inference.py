"""Run the bundled covariance examples from any working directory."""
from pathlib import Path
import argparse
import numpy as np
import torch
from pgbsa import create_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    fixture = Path(__file__).resolve().with_name("sample_covariances.npz")
    with np.load(fixture, allow_pickle=False) as data:
        x = torch.from_numpy(data["in_mat"].copy())
    with create_model(args.device) as model:
        angles = model.predict(x)
        print(f"Deployed parameters: {model.parameter_count():,}")
    for index, prediction in enumerate(angles):
        print(f"Sample {index}: {prediction.tolist()} degrees")

if __name__ == "__main__":
    main()
