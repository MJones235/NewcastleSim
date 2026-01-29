#!/usr/bin/env python3
"""
Performance and scalability testing for JuPedSim station simulation.

This script runs the simulation with varying numbers of agents (10 to 1000)
and measures execution time to analyze performance characteristics.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_simulation(num_agents: int, no_gui: bool = True) -> float:
    """
    Run the simulation with specified number of agents and measure execution time.

    Args:
        num_agents: Number of agents to simulate
        no_gui: Whether to disable GUI (default: True for benchmarking)

    Returns:
        Execution time in seconds
    """
    # Use venv python if available, otherwise use current python
    venv_python = project_root / ".venv" / "bin" / "python3"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python_exe,
        str(project_root / "run_jupedsim_station.py"),
        "--num-agents",
        str(num_agents),
    ]

    if no_gui:
        cmd.extend(["--no-gui", "--no-viz"])

    print(f"Running simulation with {num_agents} agents...", end=" ", flush=True)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=project_root  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return -1

        elapsed = time.time() - start_time
        print(f"✓ {elapsed:.2f}s")
        return elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"TIMEOUT (>{elapsed:.0f}s)")
        return -1
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: {e}")
        return -1


def run_benchmark(agent_counts: list[int]) -> dict:
    """
    Run benchmark tests for multiple agent counts.

    Args:
        agent_counts: List of agent counts to test

    Returns:
        Dictionary with results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "agent_counts": [],
        "execution_times": [],
        "failed_runs": [],
    }

    print("=" * 70)
    print("Performance Benchmark - JuPedSim Station Simulation")
    print("=" * 70)
    print()

    for count in agent_counts:
        execution_time = run_simulation(count)

        if execution_time > 0:
            results["agent_counts"].append(count)
            results["execution_times"].append(execution_time)
        else:
            results["failed_runs"].append(count)

    return results


def save_results(results: dict, filename: str = "benchmark_results.json"):
    """Save benchmark results to JSON file."""
    output_path = Path(__file__).parent / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


def plot_results(results: dict, filename: str = "performance_plot.png"):
    """
    Generate and save performance plots.

    Args:
        results: Benchmark results dictionary
        filename: Output filename for the plot
    """
    if not results["agent_counts"]:
        print("No valid results to plot.")
        return

    agent_counts = np.array(results["agent_counts"])
    execution_times = np.array(results["execution_times"])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Execution time vs number of agents
    ax1.plot(agent_counts, execution_times, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Agents", fontsize=12)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax1.set_title(
        "Simulation Performance: Execution Time vs Agent Count", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # Add trend line
    if len(agent_counts) > 1:
        z = np.polyfit(agent_counts, execution_times, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(agent_counts.min(), agent_counts.max(), 100)
        ax1.plot(x_trend, p(x_trend), "--", alpha=0.5, color="red", label="Trend (quadratic)")
        ax1.legend()

    # Plot 2: Time per agent
    time_per_agent = execution_times / agent_counts
    ax2.plot(agent_counts, time_per_agent, "s-", linewidth=2, markersize=8, color="orange")
    ax2.set_xlabel("Number of Agents", fontsize=12)
    ax2.set_ylabel("Time per Agent (seconds)", fontsize=12)
    ax2.set_title("Simulation Efficiency: Time per Agent", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to {output_path}")

    # Also display if running in interactive mode
    # plt.show()


def print_summary(results: dict):
    """Print summary statistics."""
    if not results["agent_counts"]:
        print("\nNo successful runs to summarize.")
        return

    agent_counts = np.array(results["agent_counts"])
    execution_times = np.array(results["execution_times"])

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total runs: {len(agent_counts)}")
    print(f"Agent count range: {agent_counts.min()} - {agent_counts.max()}")
    print(f"Execution time range: {execution_times.min():.2f}s - {execution_times.max():.2f}s")
    print(f"Average execution time: {execution_times.mean():.2f}s")

    if len(agent_counts) > 1:
        # Calculate scaling factor (how time increases with agents)
        time_ratio = execution_times[-1] / execution_times[0]
        agent_ratio = agent_counts[-1] / agent_counts[0]
        scaling_exponent = np.log(time_ratio) / np.log(agent_ratio)
        print(f"Scaling exponent: {scaling_exponent:.2f} (1=linear, 2=quadratic)")

    if results["failed_runs"]:
        print(f"\nFailed runs: {results['failed_runs']}")

    print("=" * 70)


def main():
    """Main entry point for performance testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance testing for JuPedSim simulation")
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        help="Specific agent counts to test (e.g., --agent-counts 10 50 100)",
    )
    parser.add_argument(
        "--min-agents", type=int, default=10, help="Minimum number of agents (default: 10)"
    )
    parser.add_argument(
        "--max-agents", type=int, default=1000, help="Maximum number of agents (default: 1000)"
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Step size for agent counts (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output filename for results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="performance_plot.png",
        help="Output filename for plot (default: performance_plot.png)",
    )

    args = parser.parse_args()

    # Determine agent counts to test
    if args.agent_counts:
        agent_counts = sorted(args.agent_counts)
    else:
        # Auto-generate reasonable test points
        if args.step:
            agent_counts = list(range(args.min_agents, args.max_agents + 1, args.step))
        else:
            # Use logarithmic spacing for better coverage
            agent_counts = [10, 20, 50, 100, 200, 500, 1000]
            agent_counts = [c for c in agent_counts if args.min_agents <= c <= args.max_agents]

    print(f"Testing with agent counts: {agent_counts}")
    print()

    # Run benchmark
    results = run_benchmark(agent_counts)

    # Save results
    save_results(results, args.output)

    # Generate plots
    plot_results(results, args.plot)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
