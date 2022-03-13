#![cfg(feature = "benchmark")]
#![feature(test)]

extern crate test;

use std::fs::File;
use std::io::{BufRead, BufReader};
use test::Bencher;
use venturi::{Network, sigmoid};

#[bench]
// Basic training benchmark. This test assumes that the working path is the
// project root that
// ```
// cargo bench --tests --features benchmark
// ```
// is run from. The training data used in this benchmark is from a wine quality
// data set rather than the mnist data set. This should not be used as an example
// of meaningful network input as the data is not prepared in any way.
fn train_100(b: &mut Bencher) {
    let mut path = std::env::current_dir().unwrap();
    path.push("tests");
    path.push("data");
    path.push("train_10_data.csv");
    let file = File::open(path).unwrap();
    let buf_reader = BufReader::new(file);
    let mut training_data: Vec<Vec<f32>> = Vec::new();
    let mut output: Vec<Vec<f32>> = Vec::new();
    let lines = buf_reader.lines();
    // pre-process the training data outside the benchmark loop
    for line in lines {
        let l = line.unwrap();
        let mut str_values: Vec<&str> = l.split(',').collect();
        // get the label as an integer to index into targets
        let label_str = str_values.pop().unwrap();
        let label = label_str.parse::<usize>().unwrap();
        let input_values: Vec<f32> = str_values.iter()
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        training_data.push(input_values);
        let mut targets = vec![0.01; 10];
        for (i, target) in targets.iter_mut().enumerate() {
            if i == label {
                *target = 0.99;
            }
        }
        output.push(targets);
    }
    let mut n = Network::new(11, 7, 10, 0.3, sigmoid);
    b.iter(|| {
        for (idx, inputs) in training_data.iter().enumerate() {
            let targets = &output[idx];
            n.train(inputs.clone(), targets.clone()).unwrap();
        }
    });
}
