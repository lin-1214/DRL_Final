import os

TEST_BENCH = "./testbenches/2025_IWLS_Contest_Benchmarks_020425/ex100.truth"
# Extract the base file name (without extension)
base_name = os.path.splitext(os.path.basename(TEST_BENCH))[0]
# Create the output string by appending '.aig'
output_file = base_name + ".aig"

print(output_file)  # Output: ex100.aig
