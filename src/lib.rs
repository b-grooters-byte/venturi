use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use ndarray::{Array, Array2};
use rand::Rng;
use std::io::{Read, Write};

const DEFAULT_LEARNING_RATE: f32 = 0.3;

type ActivationFn = fn(f32) -> f32;

/// The sigmoid activation function for neural net. The activation function
/// maps the input of a node to a range of 0.0 to 1.0. For more information,
/// see https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
pub fn sigmoid(input: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-input))
}

/// tanh is the hyperbolic activation function. tanh will map an input of a
/// node to an output range of -1.0 to 1.0. For more information see,
/// https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
pub fn tanh(input: f32) -> f32 {
    let e = std::f32::consts::E;
    let pos = e.powf(input);
    let neg = e.powf(-input);
    (pos - neg) / (pos + neg)
}

pub trait BytesSerialize {
    fn to_bytes(&self, w: &mut dyn Write);
}

pub trait BytesDeserialize {
    fn from_bytes(r: &mut dyn Read) -> Self;
}

#[derive(Debug)]
pub struct NetworkErr {
    pub message: String,
}

pub struct Network {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    learning_rate: f32,
    weights: [Array2<f32>; 2],
    activation: Option<ActivationFn>,
}

impl BytesSerialize for Network {
    fn to_bytes(&self, w: &mut dyn Write) {
        // node values
        w.write_u16::<BigEndian>(self.input_nodes as u16).unwrap();
        w.write_u16::<BigEndian>(self.hidden_nodes as u16).unwrap();
        w.write_u16::<BigEndian>(self.output_nodes as u16).unwrap();
        // learning rate
        w.write_f32::<BigEndian>(self.learning_rate).unwrap();
        // weights
        for cell in self.weights[0].iter() {
            w.write_f32::<BigEndian>(*cell).unwrap();
        }
        for cell in self.weights[1].iter() {
            w.write_f32::<BigEndian>(*cell).unwrap();
        }
    }
}

impl BytesDeserialize for Network {
    fn from_bytes(r: &mut dyn Read) -> Self {
        // read in the network shape
        let input_nodes = r.read_u16::<BigEndian>().unwrap() as usize;
        let hidden_nodes = r.read_u16::<BigEndian>().unwrap() as usize;
        let output_nodes = r.read_u16::<BigEndian>().unwrap() as usize;
        // learning rate
        let learning_rate = r.read_f32::<BigEndian>().unwrap();
        // reshape the network
        let mut weights: [Array2<f32>; 2] = [
            Array2::<f32>::zeros((hidden_nodes, input_nodes)),
            Array2::<f32>::zeros((output_nodes, hidden_nodes)),
        ];
        // set the weights
        for cell in weights[0].iter_mut() {
            *cell = r.read_f32::<BigEndian>().unwrap();
        }
        for cell in weights[1].iter_mut() {
            *cell = r.read_f32::<BigEndian>().unwrap();
        }

        Network {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            weights,
            activation: None,
        }
    }
}

impl Default for Network {
    fn default() -> Self {
        Network {
            input_nodes: 0,
            hidden_nodes: 0,
            output_nodes: 0,
            learning_rate: DEFAULT_LEARNING_RATE,
            weights: [Array2::<f32>::zeros((0, 0)), Array2::<f32>::zeros((0, 0))],
            activation: None,
        }
    }
}

impl Network {
    /// Creates a new nueral network with default configuration
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f32,
        activation: ActivationFn,
    ) -> Network {
        let mut rng = rand::thread_rng();
        let mut weights_ho = Array2::<f32>::zeros((output_nodes, hidden_nodes));
        for elem in weights_ho.iter_mut() {
            *elem = rng.gen::<f32>() - 0.5;
        }
        let mut weights_ih = Array2::<f32>::zeros((hidden_nodes, input_nodes));
        for elem in weights_ih.iter_mut() {
            *elem = rng.gen::<f32>() - 0.5;
        }
        let weights = [weights_ih, weights_ho];
        Network {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            weights,
            activation: Some(activation),
        }
    }

    pub fn set_weights(&mut self, weights: Array2<f32>, layer: usize) {
        self.weights[layer] = weights;
    }

    pub fn set_activation_fn(&mut self, activation: Option<ActivationFn>) {
        self.activation = activation;
    }

    pub fn query_hidden(&self, inputs: Vec<f32>) -> Result<Array2<f32>, NetworkErr> {
        if self.activation.is_none() {
            return Err(NetworkErr {
                message: "Activation function not set".to_string(),
            });
        }
        let input_list = Array::from_shape_vec((self.input_nodes, 1), inputs).unwrap();

        let mut hidden = self.weights[0].dot(&input_list);
        // adjust input node values by activation function
        for node in hidden.iter_mut() {
            *node = (self.activation.unwrap())(*node);
        }
        Ok(hidden)
    }

    pub fn query(&self, inputs: Vec<f32>) -> Result<Array2<f32>, NetworkErr> {
        if self.activation.is_none() {
            return Err(NetworkErr {
                message: "Activation function not set".to_string(),
            });
        }
        let activation_fn = self.activation.unwrap();
        let input_list = Array::from_shape_vec((self.input_nodes, 1), inputs).unwrap();
        let mut hidden = self.weights[0].dot(&input_list);
        // adjust input node values by activation function
        for node in hidden.iter_mut() {
            *node = activation_fn(*node);
        }
        let mut output = self.weights[1].dot(&hidden);
        // adjust output by activation function
        for node in output.iter_mut() {
            *node = activation_fn(*node);
        }
        Ok(output)
    }

    /// Train the network with the training data passed in the inputs parameter.
    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) -> Result<(), NetworkErr> {
        if self.activation.is_none() {
            return Err(NetworkErr {
                message: "Activation function not set".to_string(),
            });
        }
        let activation_fn = self.activation.unwrap();
        let input_list = Array::from_shape_vec((self.input_nodes, 1), inputs).unwrap();
        let target_list = Array::from_shape_vec((self.output_nodes, 1), targets).unwrap();
        let mut hidden = self.weights[0].dot(&input_list);
        // adjust input node values by activation function
        for node in hidden.iter_mut() {
            *node = activation_fn(*node);
        }
        let mut final_outputs = self.weights[1].dot(&hidden);
        // adjust output by activation function
        for node in final_outputs.iter_mut() {
            *node = activation_fn(*node);
        }
        // initial error
        let errors_output = target_list - &final_outputs;
        let errors_hidden = self.weights[1].t().dot(&errors_output);
        // back propagate hidden->output weights
        let error_temp = errors_output * &final_outputs * (1.0 - &final_outputs);
        let temp = self.learning_rate * error_temp.dot(&hidden.t());
        self.weights[1] = &self.weights[1] + temp;
        // back propagate input->hidden weights
        let error_temp = &errors_hidden * &hidden * (1.0 - &hidden);
        let temp = self.learning_rate * error_temp.dot(&input_list.t());
        self.weights[0] = &self.weights[0] + temp;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    ///! Tests for Network. All assert_eq statements have the parameter order:
    ///! (expected, actual)
    use ndarray::arr2;

    #[test]
    pub fn test_sigmoid() {
        let epsilon = sigmoid(10.0) - 0.999;
        assert!(epsilon < 0.001, "expected epsilon of < 0.001");
        let sigmoid = sigmoid(-10.0);
        assert!(sigmoid < 0.001, "expected sigmoid of < 0.001");
    }

    #[test]
    pub fn test_calc_hidden() {
        let test_input = vec![0.9, 0.1, 0.8];
        let test_weight = arr2(&[[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]]);
        let mut n = Network::new(3, 3, 3, 0.3, sigmoid);
        n.set_weights(test_weight, 0);
        let h = n.query_hidden(test_input).unwrap();
        assert_eq!(761, (h[[0, 0]] * 1000.0) as i32);
        assert_eq!(603, (h[[1, 0]] * 1000.0) as i32);
        assert_eq!(650, (h[[2, 0]] * 1000.0) as i32);
    }
}
