use ndarray::{Array, ShapeBuilder};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::{fmt, str::FromStr};
use structopt::StructOpt;
use venturi::{sigmoid, BytesDeserialize, BytesSerialize, Network};

#[derive(Debug)]
enum Mode {
    Train,
    Query,
}

struct ParseErr;

impl fmt::Display for ParseErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to parse Mode")
    }
}

impl fmt::Debug for ParseErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to parse Mode")
    }
}

impl FromStr for Mode {
    type Err = ParseErr;
    /// Creates a Network Mode enumeration value from a string representation
    /// A mode ParseError type is returned if the string value is not one of:
    /// * "train"
    /// * "query"
    /// this is case sensitive. The input string must be all lowercase.
    fn from_str(mode_str: &str) -> Result<Self, <Self as FromStr>::Err> {
        match mode_str {
            "train" => Ok(Mode::Train),
            "query" => Ok(Mode::Query),
            _ => Err(ParseErr {}),
        }
    }
}

#[derive(Debug, StructOpt)]
struct Options {
    #[structopt(short, long)]
    mode: Mode,
    #[structopt(short, long, required_if("mode", "query"))]
    network_file: Option<String>,
    #[structopt(short, long, required_if("mode", "query"))]
    input_file: Option<String>,
    #[structopt(short, long)]
    training_data: Option<String>,
    #[structopt(short, long)]
    output: Option<String>,
    #[structopt(short, long)]
    show_training: bool,
}

fn main() -> std::io::Result<()> {
    let opt = Options::from_args();
    println!("Venturi Nueral Netowrk");
    match opt.mode {
        Mode::Train => {
            let filename = opt.training_data.unwrap();
            println!("Training network with {}", filename);
            let file = File::open(filename)?;
            let buf_reader = BufReader::new(file);
            let lines = buf_reader.lines();
            let mut network = Network::new(784, 100, 10, 0.3, sigmoid);
            for line in lines {
                let l = line.unwrap();
                let mut str_values = l.split(',');
                // get the label as an integer to index into targets
                let label = str_values.next().unwrap().parse::<usize>().unwrap();
                let input_values: Vec<f32> = str_values
                    .map(|s| s.parse::<i32>().unwrap() as f32 / 255.0 * 0.99 + 0.01)
                    .collect();
                let mut targets = vec![0.01; 10];
                for (i, target) in targets.iter_mut().enumerate() {
                    if i == label {
                        *target = 0.99;
                    }
                }
                if let Err(e) = network.train(input_values.clone(), targets) {
                    println!("{}", e.message);
                    panic!("Unrecoverable error");
                }
                if opt.show_training {
                    let input =
                        Array::from_shape_vec((28, 28).strides((28, 1)), input_values).unwrap();
                    for row in 0..28 {
                        for col in 0..28 {
                            if input[[row, col]] > 0.65 {
                                print!("*");
                            } else if input[[row, col]] > 0.1 {
                                print!(".");
                            } else {
                                print!(" ");
                            }
                        }
                        println!();
                    }
                    println!("\n--------------------------------\n")
                }
            }
            // check for output file
            if let Some(file) = opt.output {
                let mut output = std::fs::File::create(file).expect("create failed");
                network.to_bytes(&mut output);
            }
        }
        Mode::Query => {
            // load the network
            let file = File::open(opt.network_file.unwrap())?;
            let mut buf_reader = BufReader::new(file);
            let mut network =  Network::from_bytes(&mut buf_reader);
            let file = File::open(opt.input_file.unwrap())?;
            let buf_reader = BufReader::new(file);
            let lines = buf_reader.lines();
           network.set_activation_fn(Some(sigmoid));
            // check a single handwritten number
            for line in lines {
                let l = line.unwrap();
                let mut str_values = l.split(',');
                // get the label as an integer to index into targets
                let label = str_values.next().unwrap().parse::<usize>().unwrap();
                println!("Label: {}", label);
                let input_values: Vec<f32> = str_values
                    .map(|s| s.parse::<i32>().unwrap() as f32 / 255.0 * 0.99 + 0.01)
                    .collect();
                match network.query(input_values) {
                    Ok(r) => println!("Result: \n{:?}", r),
                    Err(e) => println!("Unable to query network: {}", e.message),
                }
            }
        }
    }
    Ok(())
}
