import os
import subprocess
import tempfile
from rl_environment import ABCEnv
import numpy as np
import argparse
def run_abc_env(env, command_type, iterations=1, output_dir = "", testbench = ""):
    # Generate the resyn2 command repeated 100 times
    command = "\n".join([f"{command_type}"] * iterations)
    # Prepare output directory
    
    os.makedirs(output_dir, exist_ok=True)

    # Generate output AIG file path
    base_name = os.path.splitext(os.path.basename(testbench))[0]
    output_file = base_name + ".aig"
    temp_aig = os.path.join(output_dir, output_file)
    print(temp_aig)
    # ABC command list

    if command_type == "resyn2":
        commands = [
            f"read {env._current_aig}",
            f"&r {env._current_aig}",
            command,
            f"write {temp_aig}",
            f"print_stats"
        ]
    else:
        commands = [
            f"read {env._current_aig}",
            f"&r {env._current_aig}",
            command,
            f"write {temp_aig}",
            f"print_stats"
        ]
    
    joined = "\n".join(commands) + "\nquit\n"

    # Run ABC with subprocess
    try:
        result = subprocess.run(
            [env.abc_path],
            input=joined.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("ABC stdout:\n", result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("ABC error:\n", e.stderr.decode())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABC environment command runner")
    parser.add_argument('--testbench', required=True, help='Path to the testbench file')
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command type")

    # Subparser for deepsyn (no additional args)
    deepsyn_parser = subparsers.add_parser("deepsyn", help="Run DeepSyn command")

    # Subparser for resyn2 (allows iterations)
    resyn2_parser = subparsers.add_parser("resyn2", help="Run resyn2 command")
    resyn2_parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1,
        help="Number of resyn2 iterations to apply"
    )
    
    args = parser.parse_args()
    
    env = ABCEnv(
        abc_path="./abc/abc",
        truth_file=args.testbench,  # Use the testbench from command line
        alpha=1.0,
        beta=0.1
    )
    # Create temporary working directory
    env._temp_dir = tempfile.TemporaryDirectory()
    env._work_dir = env._temp_dir.name
    env._current_aig = os.path.join(env._work_dir, "current.aig")

    # Reset environment and write AIG
    env.reset()

    # Run ABC with heavy resyn2
    if args.command == "deepsyn":
        print("Running DeepSyn command...")
        output_dir = "./output/deepsyn"
        run_abc_env(env, "&deepsyn -v -J 10", output_dir=output_dir, testbench=args.testbench)
    elif args.command == "resyn2":
        print(f"Running resyn2 command for {args.iterations} iterations...")
        output_dir = f"./output/resyn2_{args.iterations}"
        run_abc_env(env, "resyn2", iterations=args.iterations, output_dir=output_dir, testbench=args.testbench)

    # Show result and clean up
    env.render()
    env.close()
    print("ABC environment run completed.")
