#!/bin/bash

# Quick script to collect data from launching matmul9 with a given range
# of matrix sizes. (See cache section in https://guillesanbri.com/CUDA-Benchmarks/)

# Check if enough arguments are passed
if [ $# -ne 3 ]; then
  echo "Usage: $0 <min_size> <max_size> <output_file>"
  exit 1
fi

# Get the arguments
min_size=$1
max_size=$2
output_file=$3

# Check if min_size and max_size are valid integers
if ! [[ "$min_size" =~ ^[0-9]+$ ]] || ! [[ "$max_size" =~ ^[0-9]+$ ]]; then
  echo "Error: matrix sizes must be integers."
  exit 1
fi

# Initialize the CSV file and write the header
echo "matrix_size,a,b" > "$output_file"

# Calculate the number of iterations
total_iterations=$(( (max_size - min_size) / 32 + 1 ))
counter=0

# Loop over the matrix sizes starting from min_size to max_size, incrementing by 32
for size in $(seq $min_size 32 $max_size); do
  # Call the matmul9 script and capture the output
  result=$(./../matmul9 $size)

  # Parse the result, assuming 'a' is on the first line and 'b' is on the second line
  a=$(echo "$result" | sed -n '1p')  # First line (a)
  b=$(echo "$result" | sed -n '2p')  # Second line (b)

  # Write the matrix size, a, and b to the CSV file
  echo "$size,$a,$b" >> "$output_file"

  # Update the counter and show progress every 5 iterations
  ((counter++))
  if (( counter % 5 == 0 )); then
    progress=$((counter * 100 / total_iterations))
    echo "Progress: $progress% ($counter/$total_iterations iterations)"
  fi
done

echo "Results saved to $output_file."
