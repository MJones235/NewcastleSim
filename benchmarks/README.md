# Performance Benchmarking

This directory contains tools for testing the performance and scalability of the JuPedSim station simulation.

## Quick Start

Run the default benchmark (tests 10, 20, 50, 100, 200, 500, 1000 agents):

```bash
python3 benchmarks/performance_test.py
```

## Usage

### Basic Usage

```bash
# Run with default settings (10 to 1000 agents)
python3 benchmarks/performance_test.py

# Specify custom range
python3 benchmarks/performance_test.py --min-agents 10 --max-agents 500

# Specify exact agent counts to test
python3 benchmarks/performance_test.py --agent-counts 10 50 100 250 500

# Use custom step size
python3 benchmarks/performance_test.py --min-agents 10 --max-agents 200 --step 20
```

### Output Options

```bash
# Custom output filenames
python3 benchmarks/performance_test.py --output my_results.json --plot my_plot.png
```

## Output Files

The benchmark script generates:

1. **benchmark_results.json** - Raw performance data including:
   - Timestamp
   - Agent counts tested
   - Execution times
   - Failed runs (if any)

2. **performance_plot.png** - Visualization showing:
   - Execution time vs number of agents
   - Time per agent (efficiency metric)
   - Trend lines

## Interpreting Results

### Scaling Exponent

The scaling exponent indicates how execution time grows with agent count:
- **~1.0**: Linear scaling (ideal)
- **~2.0**: Quadratic scaling (common for many-body simulations)
- **>2.0**: Super-quadratic scaling (potential optimization needed)

### Time per Agent

This metric shows the average time spent per agent:
- Flat line: Good scalability
- Increasing: Overhead grows with agent count

## Example Results

After running the benchmark, you'll see output like:

```
======================================================================
Performance Benchmark - JuPedSim Station Simulation
======================================================================

Running simulation with 10 agents... ✓ 1.23s
Running simulation with 20 agents... ✓ 1.45s
Running simulation with 50 agents... ✓ 2.34s
...

======================================================================
SUMMARY STATISTICS
======================================================================
Total runs: 7
Agent count range: 10 - 1000
Execution time range: 1.23s - 45.67s
Average execution time: 12.34s
Scaling exponent: 1.89 (1=linear, 2=quadratic)
======================================================================
```

## Notes

- GUI is automatically disabled during benchmarks for consistent timing
- Each simulation has a 5-minute timeout
- Failed runs are logged but don't stop the benchmark
- Results are cumulative - you can run multiple benchmarks and compare
