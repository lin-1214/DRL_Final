#!/bin/bash

# Directories containing the benchmark files
IWLS_DIR="./testbenches/2025_IWLS_Contest_Benchmarks_020425"
EPFL_DIR="./testbenches/EPFL"

# Function to run experiments for a single benchmark
run_benchmark() {
    local benchmark=$1
    echo "Processing benchmark: $benchmark"

    # Run default commands
    echo "Running DeepSyn..."
    python3 run_default_commands.py --testbench "$benchmark" deepsyn

    echo "Running Resyn2..."
    python3 run_default_commands.py --testbench "$benchmark" resyn2 --iterations 10000

    # Run RL environment
    echo "Running MCTS optimization..."
    python3 rl_environment.py --testbench "$benchmark"


    echo "Completed processing $benchmark"
    echo "----------------------------------------"
}

# Specific benchmarks to process
BENCHMARKS=(
    "$IWLS_DIR/ex100.truth"
    # "$IWLS_DIR/ex104.truth"
    # "$IWLS_DIR/ex108.truth"
    # "$IWLS_DIR/ex112.truth"
    # "$EPFL_DIR/adder.truth"
    # "$EPFL_DIR/bar.truth"
    # "$EPFL_DIR/max.truth"
    # "$EPFL_DIR/sin.truth"
)

# Process specified benchmarks
for benchmark in "${BENCHMARKS[@]}"; do
    if [ -f "$benchmark" ]; then
        run_benchmark "$benchmark"
    else
        echo "Warning: Benchmark file not found: $benchmark"
    fi
done

echo "All benchmarks processed successfully!"