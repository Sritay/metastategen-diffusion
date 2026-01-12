
def read_file(path, n_lines=None):
    print(f"\n--- Reading {path} ---")
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            if n_lines and n_lines < len(lines):
                print(f"(Last {n_lines} lines)")
                lines = lines[-n_lines:]
            for line in lines:
                print(line.strip())
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    read_file("runs/day11_al_23_hpc/al_metrics.csv")
    read_file("slurm-90-12158761.out", n_lines=50) # Last 50 lines
