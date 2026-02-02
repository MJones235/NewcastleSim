#!/usr/bin/env python3
"""
Profile the Station Concordia simulation to identify performance bottlenecks.

Usage:
    python tools/profile_concordia.py [--no-llm] [--max-decisions N]
"""

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Profile Station Concordia simulation")
    parser.add_argument("--no-llm", action="store_true", help="Use mock LLM")
    parser.add_argument("--max-decisions", type=int, default=3, help="Maximum decisions to profile")
    parser.add_argument(
        "--config", type=str, default="scenarios/station_concordia/config/config.yaml"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Station Concordia Performance Profiler")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Max decisions: {args.max_decisions}")
    print(f"LLM: {'Mock' if args.no_llm else 'Azure OpenAI'}")
    print("=" * 70)
    print()

    # Import here to avoid slowdown in arg parsing
    from scenarios.station_concordia.run_station_concordia import (
        load_config,
        run_simulation,
        setup_language_model,
    )

    # Load config
    config = load_config(args.config)

    # Override max_steps to limit profiling time
    # Each decision happens every 5 seconds, so 100 steps = 5 seconds of sim time
    config["simulation"]["max_iterations"] = args.max_decisions * 100

    # Setup model
    print("Setting up language model...")
    model, embedder = setup_language_model(config, disable_llm=args.no_llm)
    print("Model ready.\n")

    # Profile the simulation
    print("Starting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        results = run_simulation(config, model, embedder)
    except Exception as e:
        print(f"\nSimulation error: {e}")
        import traceback

        traceback.print_exc()

    profiler.disable()

    print("\n" + "=" * 70)
    print("Profiling Results")
    print("=" * 70)

    # Print timing summary from results
    if "results" in locals():
        print("\nSimulation completed:")
        print(f"  Steps: {results.get('steps', 0)}")
        print(f"  Sim time: {results.get('sim_time', 0):.1f}s")
        print(f"  Real time: {results.get('elapsed_time', 0):.1f}s")
        print(f"  Decisions: {results.get('decisions_made', 0)}")
        print(f"  Events: {results.get('events_triggered', 0)}")

    # Print profiling stats
    print("\n" + "=" * 70)
    print("Top 30 Functions by Cumulative Time")
    print("=" * 70)

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())

    print("\n" + "=" * 70)
    print("Top 30 Functions by Total Time (excluding subcalls)")
    print("=" * 70)

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("time")
    stats.print_stats(30)
    print(stream.getvalue())

    # Save detailed stats to file
    output_file = Path("scenarios/station_concordia/output/profile_stats.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        f.write("=" * 70 + "\n")
        f.write("CUMULATIVE TIME\n")
        f.write("=" * 70 + "\n")
        stats.print_stats()

        f.write("\n" + "=" * 70 + "\n")
        f.write("TOTAL TIME\n")
        f.write("=" * 70 + "\n")
        stats.sort_stats("time")
        stats.print_stats()

    print(f"\nDetailed stats saved to: {output_file}")
    print("\nTo analyze further:")
    print(f"  python -m pstats {output_file}")


if __name__ == "__main__":
    main()
