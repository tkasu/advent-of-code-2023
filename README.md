# My Advent of Code 2023 Solutions

## Requirements

* https://rustup.rs/
* (Optional to use CUDA): https://developer.nvidia.com/cuda-toolkit

## Running the solutions

* `cargo run -r`
* With cuda: `cargo run -r --features cuda`

## Building CUDA ptx file

The prebuilt .ptx is included in the repo, but if you want to rebuild it, you can use the following command:

`nvcc -ptx src/day5_kernel.cu && mv day5_kernel.ptx src/`
