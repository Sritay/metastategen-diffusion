
import matplotlib.pyplot as plt
import re
import argparse

def parse_log(log_path):
    steps = []
    energies = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Pattern: "Step {k}: E_norm={e} | ..."
            # Regex to capture k and e
            match = re.search(r"Step (\d+): E_norm=([\d\.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                energies.append(float(match.group(2)))
                
    return steps, energies

def main():
    steps, energies = parse_log("logs/refinement_trace_shake_200k.txt")
    
    if not steps:
        print("No data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, energies, label='Energy Norm (Sample 1)')
    plt.xlabel('Refinement Step')
    plt.ylabel('Energy (Normalized)')
    plt.title('Refinement Convergence (SHAKE Enabled, 200k Steps)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = "logs/refinement_energy_plot.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
