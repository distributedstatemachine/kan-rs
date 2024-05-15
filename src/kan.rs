use plotpy::{Plot, Scatter};
use tch::{nn, Tensor};
use crate::symbolic_kan_layer::SymbolicKANLayer;
use crate::kan_layer::KANLayer;


/// Represents the Kolmogorov-Arnold Network (KAN) model.
///
/// The `KAN` struct contains the layers and configurations of the KAN model.
///
/// # Fields
///
/// * `biases` - The bias terms for each layer, represented as `nn::Linear` modules.
/// * `act_fun` - The activation functions for each layer, represented as `KANLayer` instances.
/// * `symbolic_fun` - The symbolic representation of the activation functions for each layer, represented as `SymbolicKANLayer` instances.
/// * `depth` - The depth of the KAN model (number of layers).
/// * `width` - The width of each layer in the KAN model.
/// * `grid` - The number of grid intervals.
/// * `k` - The order of the piecewise polynomial.
/// * `base_fun` - The residual function b(x) as an `nn::Module`.
/// * `symbolic_enabled` - Indicates whether symbolic computation is enabled.
/// * `device` - The device on which the model is running (e.g., CPU or GPU).
struct KAN {
    biases: Vec<nn::Linear>,
    act_fun: Vec<KANLayer>,
    symbolic_fun: Vec<Symbolic_KANLayer>,
    depth: usize,
    width: Vec<usize>,
    grid: usize,
    k: usize,
    base_fun: nn::Module,
    symbolic_enabled: bool,
    device: tch::Device,
}

impl KAN {
    fn new(
        width: &[usize],
        grid: usize,
        k: usize,
        base_fun: nn::Module,
        symbolic_enabled: bool,
        device: tch::Device,
    ) -> Self {
        let depth = width.len() - 1;
        let mut biases = Vec::with_capacity(depth);
        let mut act_fun = Vec::with_capacity(depth);
        let mut symbolic_fun = Vec::with_capacity(depth);

        for l in 0..depth {
            // Create biases
            let bias = nn::linear(width[l + 1], 1, Default::default(), Default::default());
            biases.push(bias);

            // Create activation functions
            let scale_base = Tensor::ones(&[width[l] * width[l + 1]], (Kind::Float, device));
            let scale_sp = Tensor::ones(&[width[l] * width[l + 1]], (Kind::Float, device));
            let coef = Tensor::zeros(&[width[l + 1], width[l], k + 1], (Kind::Float, device));
            let mask = Tensor::ones(&[width[l + 1] * width[l]], (Kind::Float, device));
            let kan_layer = KANLayer {
                in_dim: width[l],
                out_dim: width[l + 1],
                num: grid,
                k,
                scale_base,
                scale_sp,
                base_fun: base_fun.clone(),
                coef,
                mask,
                grid: Tensor::linspace(0.0, 1.0, grid + 1, (Kind::Float, device)),
                sp_trainable: true,
                sb_trainable: true,
            };
            act_fun.push(kan_layer);

            // Initialize symbolic functions
            let symbolic_layer = Symbolic_KANLayer {
                in_dim: width[l],
                out_dim: width[l + 1],
                mask: Tensor::ones(&[width[l + 1], width[l]], (Kind::Float, device)),
                affine: Tensor::zeros(&[width[l + 1], width[l], 4], (Kind::Float, device)),
                funs_sympy: vec![vec!["".to_string(); width[l]]; width[l + 1]],
            };
            symbolic_fun.push(symbolic_layer);
        }

        KAN {
            biases,
            act_fun,
            symbolic_fun,
            depth,
            width: width.to_vec(),
            grid,
            k,
            base_fun,
            symbolic_enabled,
            device,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut acts = vec![x.shallow_clone()];
        let mut spline_preacts = Vec::with_capacity(self.depth);
        let mut spline_postacts = Vec::with_capacity(self.depth);
        let mut spline_postsplines = Vec::with_capacity(self.depth);
        let mut acts_scale = Vec::with_capacity(self.depth);
        let mut acts_scale_std = Vec::with_capacity(self.depth);

        for l in 0..self.depth {
            let (x_numerical, preacts, postacts_numerical, postspline) =
                self.act_fun[l].forward(&x);
            let (x_symbolic, postacts_symbolic) = if self.symbolic_enabled {
                self.symbolic_fun[l].forward(&x)
            } else {
                (
                    Tensor::zeros_like(&x_numerical),
                    Tensor::zeros_like(&postacts_numerical),
                )
            };

            let x = &x_numerical + &x_symbolic;
            let postacts = &postacts_numerical + &postacts_symbolic;

            spline_preacts.push(preacts);
            spline_postacts.push(postacts.shallow_clone());
            spline_postsplines.push(postspline);

            let grid_reshape = self.act_fun[l]
                .grid
                .view([self.width[l + 1], self.width[l], -1]);
            let input_range = grid_reshape.i((.., .., -1)) - grid_reshape.i((.., .., 0)) + 1e-4;
            let output_range = postacts.mean_dim([0], false, Kind::Float);
            acts_scale.push(output_range / input_range);
            acts_scale_std.push(postacts.std_dim([0], false, Kind::Float));

            let x = &x + &self.biases[l].ws;
            acts.push(x.shallow_clone());
        }

        acts.last().unwrap().shallow_clone()
    }
}