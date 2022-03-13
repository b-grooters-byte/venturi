![Build](https://github.com/bytetrail/venturi/actions/workflows/rust-build/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
# Venturi
Venturi is a lightweight, configurable neural net library developed using Rust. 
The library has been developed to provide a starting point for adding basic AI
capabilities to low-cost embedded systems.

It was originally inspired by the neural net code presented in the book, "Make Your
Own Neural Network" by Tariq Rashid. 

The MNIST (Modified National Institute of Technology database) database is used
both in the book and to exercise the original Rust implementation of the neural
net. Small samples of the MNIST database are included that can be used both
for training and query experiments.

# Benchmark
The project includes a simple baseline benchmark of the training method using a 
fixed dataset. The benchmark feature needs to be enabled to run the tests and 
you must use the nightly toolchain:

    % rustup override set nightly
    % cargo bench --tests --features benchmark

### Benchmark Baselines
The benchmark baselines are taken from the average ns/iter ovber 5 benchmark runs on each platform.

| Platform             | OS                    | CPU                  | Memory                 | Benchmark               |
|----------------------|-----------------------|----------------------|------------------------|-------------------------|
| MacBook Pro 15" 2019 | 12.2.1                | 2.6GHz Intel Core i7 | 16GB 2400 MHz DDR4     | 26_772 ns/iter +/- 1661 |
| Raspberry Pi 4       | Raspbian/GNU/Linux 11 | Cortex-A72 (ARM v8) 64-bit 1.5GHz | 4GB LPDDR4-3200 SDRAM  | 72_602 ns/iter +/- 1190 |


# Command Line Interface
The project includes a CLI entry point that is able to exercise the library 
capabilities.

You can see the command line help by running venturi either through the cargo 
run command

    cargo run --release -- --help

or by running venturi directly 

    venturi --help

The mode argument must be one of:
 * train 
 * query 

The input-file argument is used in query mode and the training-data argument is
used in training mode. 

The output-file argument may be used to create a binary file of the trained
network for future queries.

```
USAGE:
venturi [FLAGS] [OPTIONS] --mode <mode>

FLAGS:
-h, --help             Prints help information
-s, --show-training    
-V, --version          Prints version information

OPTIONS:
    -H, --hidden-node-count <hidden-node-count>    
    -i, --input-file <input-file>                  
    -I, --input-node-count <input-node-count>      
    -m, --mode <mode>                              
    -n, --network-file <network-file>              
    -o, --output <output>                          
    -O, --output-node-count <output-node-count>    
    -t, --training-data <training-data> 
```
