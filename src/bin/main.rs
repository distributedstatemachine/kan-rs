use plotters::prelude::*;
use rand::prelude::*;
use tch::{nn, Device, Tensor};

use kan::{create_dataset, KAN};

fn main() {
    let device = Device::cuda_if_available();
    let model = KAN::new(&[2, 5, 1], 5, 3, nn::activation::Tanh, true, device);

    let f =
        |x: &Tensor| x.slice(0, 0, 1, 1).sin() * std::f64::consts::PI + x.slice(0, 1, 2, 1).pow(2);
    let dataset = create_dataset(&f, 2);

    // Plot the KAN at initialization
    model.forward(&dataset["train_input"]);
    model.plot("figures", 100);

    // Train the KAN with sparsity regularization
    model.train(
        &dataset,
        "LBFGS",
        20,
        1,
        0.01,
        10.0,
        0.0,
        0.0,
        true,
        10,
        None,
        1.0,
        50,
        -1,
        1e-16,
        1.0,
        &[],
        false,
        false,
        &vec![],
        &vec![],
        3,
        1,
        1.0,
        "video",
        device,
    );

    // Plot the trained KAN
    model.plot("figures", 100);

    // Prune the KAN and replot (keep the original shape)
    model.prune(0.01);
    model.plot("figures", 100);

    // Prune the KAN and replot (get a smaller shape)
    let model = model.prune(0.01);
    model.forward(&dataset["train_input"]);
    model.plot("figures", 100);

    // Continue training and replot
    model.train(
        &dataset,
        "LBFGS",
        20,
        1,
        0.01,
        10.0,
        0.0,
        0.0,
        true,
        10,
        None,
        1.0,
        50,
        -1,
        1e-16,
        1.0,
        &[],
        false,
        false,
        true,
        &vec![],
        3,
        1,
        1.0,
        "video",
        device,
    );
    model.plot("figures", 100);

    // Automatically or manually set activation functions to be symbolic
    let mode = "auto"; // or "manual"

    if mode == "manual" {
        model.fix_symbolic(0, 0, 0, "sin", true);
        model.fix_symbolic(0, 1, 0, "x^2", true);
        model.fix_symbolic(1, 0, 0, "exp", true);
    } else if mode == "auto" {
        let lib = vec![
            "x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs",
        ];
        model.auto_symbolic((-10.0, 10.0), (-10.0, 10.0), &lib, 1);
    }

    // Continue training to almost machine precision
    model.train(
        &dataset,
        "LBFGS",
        50,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        false,
        0,
        None,
        1.0,
        50,
        -1,
        1e-16,
        1.0,
        &[],
        false,
        false,
        false,
        &vec![],
        &vec![],
        3,
        1,
        "video",
        device,
    );

    // Obtain the symbolic formula
    let (formula, vars) = model.symbolic_formula(
        2,
        &vec!["x_1".to_string(), "x_2".to_string()],
        None,
        false,
        None,
    );
    println!("Symbolic formula: {:?}", formula);
}
