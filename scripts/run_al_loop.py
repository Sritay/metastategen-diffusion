#!/usr/bin/env python
import argparse
import sys
from metastategen.workflows.active_learning import run_active_learning

def main():
    parser = argparse.ArgumentParser(description="Proxy script for active learning loop")
    parser.add_argument("--config", type=str, default="configs/ala2_al.yaml")
    args = parser.parse_args()
    
    return run_active_learning(config_path=args.config)

if __name__ == "__main__":
    sys.exit(main())
