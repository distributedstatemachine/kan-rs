use tch::{Tensor, Device};
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

mod kan;
use kan::{KAN, StateDict};

fn main() {
    let device = Device::cuda_if_available();

    // Define the model architecture
    let width = vec![2, 5, 1];
    let grid = 5;
    let k = 3;
    let base_fun = nn::seq(&[
        nn::linear(2, 10, Default::default()),
        nn::relu(),
        nn::linear(10, 1, Default::default()),
    ]);
    let symbolic_enabled = true;

    // Create a new KAN model
    let mut model = KAN::new(&width, grid, k, base_fun, symbolic_enabled, device);

    // Prepare the dataset
    let (train_input, train_label) = prepare_train_data();
    let (test_input, test_label) = prepare_test_data();

    // Define the training parameters
    let epochs = 100;
    let learning_rate = 0.01;
    let optimizer = nn::Adam::default().build(&model.parameters(), learning_rate).unwrap();

    // Train the model
    for epoch in 1..=epochs {
        let loss = model.train(&train_input, &train_label, &optimizer);
        println!("Epoch: {}, Loss: {}", epoch, loss);

        // Evaluate the model on the test set
        let test_loss = model.evaluate(&test_input, &test_label);
        println!("Test Loss: {}", test_loss);
    }

    // Prune the model
    let pruned_model = model.prune(0.01);

    // Perform symbolic fitting
    let symbolic_lib = vec!["sin", "cos", "exp"];
    pruned_model.auto_symbolic(&symbolic_lib);

    // Visualize the model
    pruned_model.visualize();

    // Save the model checkpoint
    let state_dict = pruned_model.state_dict();
    save_checkpoint(&state_dict, "kan_model.pt");
}

fn prepare_train_data() -> (Tensor, Tensor) {
    // Load train input data from file
    let train_input_file = File::open("train_input.txt").expect("Failed to open train input file");
    let train_input_reader = std::io::BufReader::new(train_input_file);
    let train_input_data: Vec<f64> = train_input_reader
        .lines()
        .map(|line| line.expect("Failed to read line").parse().expect("Failed to parse number"))
        .collect();
    let train_input = Tensor::of_slice(&train_input_data).view([-1, 2]);

    // Load train label data from file
    let train_label_file = File::open("train_label.txt").expect("Failed to open train label file");
    let train_label_reader = std::io::BufReader::new(train_label_file);
    let train_label_data: Vec<f64> = train_label_reader
        .lines()
        .map(|line| line.expect("Failed to read line").parse().expect("Failed to parse number"))
        .collect();
    let train_label = Tensor::of_slice(&train_label_data).view([-1, 1]);

    (train_input, train_label)
}

fn prepare_test_data() -> (Tensor, Tensor) {
    // Load test input data from file
    let test_input_file = File::open("test_input.txt").expect("Failed to open test input file");
    let test_input_reader = std::io::BufReader::new(test_input_file);
    let test_input_data: Vec<f64> = test_input_reader
        .lines()
        .map(|line| line.expect("Failed to read line").parse().expect("Failed to parse number"))
        .collect();
    let test_input = Tensor::of_slice(&test_input_data).view([-1, 2]);

    // Load test label data from file
    let test_label_file = File::open("test_label.txt").expect("Failed to open test label file");
    let test_label_reader = std::io::BufReader::new(test_label_file);
    let test_label_data: Vec<f64> = test_label_reader
        .lines()
        .map(|line| line.expect("Failed to read line").parse().expect("Failed to parse number"))
        .collect();
    let test_label = Tensor::of_slice(&test_label_data).view([-1, 1]);

    (test_input, test_label)
}

fn save_checkpoint(state_dict: &[(&str, Tensor)], path: &str) {
    let mut file = File::create(path).expect("Failed to create checkpoint file");
    for (name, tensor) in state_dict {
        let data = tensor.data();
        file.write_all(format!("{}:{:?}\n", name, data).as_bytes())
            .expect("Failed to write to checkpoint file");
    }
}