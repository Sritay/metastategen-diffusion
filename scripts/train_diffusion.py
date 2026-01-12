#!/usr/bin/env python
import argparse
import sys
from metastategen.workflows.train import run_training

def main():
    parser = argparse.ArgumentParser(description="Proxy script for training diffusion model")
    parser.add_argument("--config", type=str, default="configs/ala2_default.yaml")
    args = parser.parse_args()
    
    sys.exit(run_training(config_path=args.config))

if __name__ == "__main__":
    main()
