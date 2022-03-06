
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
-i, --input-file <input-file>          
-m, --mode <mode>                      
-n, --network-file <network-file>      
-o, --output <output>                  
-t, --training-data <training-data>    
```
