# DRL_Final
Final project for Deep Reinforcement Learning 2025@NTU

## 📁 Project Structure

    ├── abc/                         # Cloned ABC logic synthesis tool
    ├── abc.history                  # ABC command history (optional)
    ├── helper_functions.py          # Internal utility functions (⚠️ Do not use directly)
    ├── requirements.txt             # Python dependencies
    ├── rl_environment.py            # Main environment entry point (✅ use this)
    ├── runtime_results/             # Output directory
    │   └── current.aig              # Example AIG file
    └── testbenches/                 # Benchmark circuits
        ├── 2025_IWLS_Contest_Benchmarks_020425/
        └── EPFL/

## Getting Started

> ⚠️ **Important**: All commands below should be executed from the root `DRL_Final` directory.

### 1. Clone the ABC Library

You must clone and build the ABC logic synthesis tool in the `abc/` folder:

    git clone https://github.com/berkeley-abc/abc.git abc
    cd abc
    make
    cd ..

### 2. Install Python Dependencies

Use the provided `requirements.txt` to install the required packages:

    pip install -r requirements.txt

### 3. Run the Environment

Several scripts are provided to run different optimization approaches:

- Run MCTS-based optimization only:
    ```bash
    ./scripts/run_mcts.sh
    ```

- Run DeepSyn optimization only:
    ```bash
    ./scripts/run_deepsyn.sh
    ```

- Run Resyn2 optimization only:
    ```bash
    ./scripts/run_resyn2.sh
    ```

- Run all optimization methods sequentially:
    ```bash
    ./scripts/run_all.sh
    ```

By default, these scripts will process the benchmark `ex100.truth`. To process additional benchmarks, modify the `BENCHMARKS` array in the script by uncommenting the desired benchmark files:

```bash
BENCHMARKS=(
    "$IWLS_DIR/ex100.truth" # Default
    # "$IWLS_DIR/ex104.truth"
    # "$IWLS_DIR/ex108.truth"
    # "$IWLS_DIR/ex112.truth"
    "$EPFL_DIR/adder.truth"  # Uncomment to process adder.truth
    # "$EPFL_DIR/bar.truth"
    # "$EPFL_DIR/max.truth"
    # "$EPFL_DIR/sin.truth"
)
```

The script will then process each uncommented benchmark sequentially through all optimization methods.




