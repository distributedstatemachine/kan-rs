use plotpy::{Plot, Scatter};
use tch::{nn, Tensor};

/// Represents a single layer in the Kolmogorov-Arnold Network (KAN).
///
/// The `KANLayer` struct contains the necessary parameters and configurations
/// for a single layer in the KAN model.
///
/// # Fields
///
/// * `in_dim` - The input dimension of the layer.
/// * `out_dim` - The output dimension of the layer.
/// * `num` - The number of grid intervals.
/// * `k` - The order of the piecewise polynomial.
/// * `scale_base` - The scale of the residual function b(x).
/// * `scale_sp` - The scale of the base function spline(x).
/// * `base_fun` - The residual function b(x) as an `nn::Module`.
/// * `coef` - The coefficients of the B-spline bases.
/// * `mask` - The mask of spline functions. Setting an element to zero means setting the corresponding activation to zero function.
/// * `grid` - The grid points.
/// * `sp_trainable` - Indicates whether `scale_sp` is trainable.
/// * `sb_trainable` - Indicates whether `scale_base` is trainable.
struct KANLayer {
    in_dim: usize,
    out_dim: usize,
    num: usize,
    k: usize,
    scale_base: Tensor,
    scale_sp: Tensor,
    base_fun: nn::Module,
    coef: Tensor,
    mask: Tensor,
    grid: Tensor,
    sp_trainable: bool,
    sb_trainable: bool,
}

impl KANLayer {
    fn new(
        in_dim: usize,
        out_dim: usize,
        num: usize,
        k: usize,
        scale_base: Tensor,
        scale_sp: Tensor,
        base_fun: nn::Module,
        coef: Tensor,
        mask: Tensor,
        grid: Tensor,
        sp_trainable: bool,
        sb_trainable: bool,
    ) -> Self {
        KANLayer {
            in_dim,
            out_dim,
            num,
            k,
            scale_base,
            scale_sp,
            base_fun,
            coef,
            mask,
            grid,
            sp_trainable,
            sb_trainable,
        }
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        let batch_size = x.size()[0];
        let x_expand = x
            .view([batch_size, self.in_dim, 1])
            .expand([batch_size, self.in_dim, self.out_dim], false);
        let preacts = x_expand * self.mask.view([1, self.in_dim, self.out_dim]);

        let grid_expand = self.grid.view([1, 1, self.num + 1]).expand(
            [batch_size, self.in_dim * self.out_dim, self.num + 1],
            false,
        );
        let coef_expand = self.coef.view([self.in_dim * self.out_dim, self.k + 1, 1]);
        let basis = polynomial_basis(
            &preacts.view([batch_size, self.in_dim * self.out_dim, 1]),
            &grid_expand,
            self.k,
        );
        let spline = basis
            .matmul(&coef_expand)
            .view([batch_size, self.out_dim, self.in_dim]);

        let base_out = self
            .base_fun
            .forward(&preacts.view([batch_size * self.in_dim * self.out_dim]))
            .view([batch_size, self.out_dim, self.in_dim]);
        let postacts = (&base_out * &self.scale_base.view([1, self.out_dim, self.in_dim]))
            + (&spline * &self.scale_sp.view([1, self.out_dim, self.in_dim]));

        let postacts_numerical = postacts
            .sum_dim_intlist(&[-1], false, Kind::Float)
            .view([batch_size, self.out_dim]);

        (
            preacts,
            postacts_numerical,
            spline.view([batch_size, self.out_dim, self.in_dim]),
            postacts,
        )
    }

    fn update_grid(&mut self, samples: &Tensor) {
        let min = samples.min().unwrap();
        let max = samples.max().unwrap();
        self.grid = Tensor::linspace(min, max, self.num + 1, (Kind::Float, Device::Cpu));
    }

    fn get_subset(&self, active_in: &[usize], active_out: &[usize]) -> Self {
        let sub_in_dim = active_in.len();
        let sub_out_dim = active_out.len();
        let sub_scale_base = self
            .scale_base
            .index_select(0, &Tensor::of_slice(active_out))
            .view([sub_out_dim * sub_in_dim]);
        let sub_scale_sp = self
            .scale_sp
            .index_select(0, &Tensor::of_slice(active_out))
            .view([sub_out_dim * sub_in_dim]);
        let sub_coef = self
            .coef
            .index_select(0, &Tensor::of_slice(active_out))
            .index_select(1, &Tensor::of_slice(active_in));
        let sub_mask = self
            .mask
            .index_select(0, &Tensor::of_slice(active_out))
            .view([sub_out_dim * sub_in_dim]);

        KANLayer {
            in_dim: sub_in_dim,
            out_dim: sub_out_dim,
            num: self.num,
            k: self.k,
            scale_base: sub_scale_base,
            scale_sp: sub_scale_sp,
            base_fun: self.base_fun.clone(),
            coef: sub_coef,
            mask: sub_mask,
            grid: self.grid.copy(),
            sp_trainable: self.sp_trainable,
            sb_trainable: self.sb_trainable,
        }
    }

    fn train(&mut self, x: &Tensor, y: &Tensor, lr: f64, optimizer: &mut nn::Optimizer) {
        let (_, _, _, postacts) = self.forward(x);
        let loss = postacts.mse_loss(y, tch::Reduction::Mean);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    fn update_coef(&mut self, x: &Tensor, y: &Tensor) {
        let (preacts, _, _, _) = self.forward(x);
        let grid_expand = self.grid.view([1, 1, self.num + 1]).expand(
            [x.size()[0], self.in_dim * self.out_dim, self.num + 1],
            false,
        );
        let basis = polynomial_basis(
            &preacts.view([x.size()[0], self.in_dim * self.out_dim, 1]),
            &grid_expand,
            self.k,
        );

        let target = y.view([x.size()[0], self.out_dim, self.in_dim]);
        let basis_t = basis.transpose(1, 2);
        let gram = basis_t.matmul(&basis);
        let coef_new = gram
            // TODO: placeholder , find more suitable value for rcond.
            .pinverse(1e-15)
            .matmul(&basis_t)
            .matmul(&target)
            .view_as(&self.coef);

        self.coef.copy_(&coef_new);
    }

    fn lock(&mut self, ids: &[(usize, usize)]) {
        let mut unique_ids = ids.to_vec();
        unique_ids.sort_unstable();
        unique_ids.dedup();

        for &(i, j) in &unique_ids {
            let index = j * self.in_dim + i;
            let target_index = ids[0].1 * self.in_dim + ids[0].0;
            self.coef
                .index_mut(index)
                .copy_(&self.coef.index(target_index));
        }
    }

    fn lock_column(&mut self, column_indices: &[usize]) {
        for j in column_indices {
            for i in 0..self.in_dim {
                let index = *j * self.in_dim + i;
                let target_index = column_indices[0] * self.in_dim + i;
                self.coef
                    .index_mut(index)
                    .copy_(&self.coef.index(target_index));
            }
        }
    }

    fn prune(&mut self, threshold: f64) {
        let mask = self.mask.view([self.out_dim, self.in_dim]);
        let scale = self.scale_base.view([self.out_dim, self.in_dim])
            + self.scale_sp.view([self.out_dim, self.in_dim]);
        let importance = mask * scale;
        let pruned_mask = importance.gt(threshold).to_kind(Kind::Float);

        self.mask.copy_(&pruned_mask.view([-1]));
    }

    fn initialize_grid_from_parent(&mut self, parent: &KANLayer, activations: &Tensor) {
        let min_val = activations.min().unwrap();
        let max_val = activations.max().unwrap();
        let mut new_grid =
            Tensor::linspace(min_val, max_val, self.num + 1, (Kind::Float, Device::Cpu));

        let old_grid = parent.grid.to_device(Device::Cpu);
        let old_min = old_grid.min().unwrap();
        let old_max = old_grid.max().unwrap();
        new_grid = (new_grid - min_val) / (max_val - min_val) * (old_max - old_min) + old_min;

        self.grid.copy_(&new_grid.to_device(self.grid.device()));
    }

    fn update_grid_from_samples(&mut self, samples: &Tensor) {
        let bounds = samples.slice(
            0,
            0,
            samples.size()[0],
            (samples.size()[0] as f64 * self.grid_eps) as i64,
        );
        let min_val = bounds.min().unwrap();
        let max_val = bounds.max().unwrap();
        self.grid = Tensor::linspace(
            min_val,
            max_val,
            self.num + 1,
            (Kind::Float, self.grid.device()),
        );
    }

    fn init_grid_params(&mut self, eps: f64, grid_range: &[f64; 2], device: Device) {
        self.grid_eps = eps;
        let min_val = grid_range[0];
        let max_val = grid_range[1];
        self.grid = Tensor::linspace(min_val, max_val, self.num + 1, (Kind::Float, device));
    }

    fn set_params(&mut self, params: &[f64]) {
        let total_params = self.in_dim * self.out_dim * (self.k + 1);
        assert_eq!(
            params.len(),
            total_params,
            "Number of parameters does not match"
        );

        let new_coef = Tensor::of_slice(params)
            .view([self.out_dim, self.in_dim, self.k + 1])
            .to_device(self.coef.device());
        self.coef.copy_(&new_coef);
    }

    fn get_params(&self) -> Vec<f64> {
        self.coef.copy().view([-1]).into()
    }
}

fn polynomial_basis(x: &Tensor, grid: &Tensor, k: usize) -> Tensor {
    let mut basis = vec![Tensor::ones_like(x)];
    for i in 1..=k {
        basis.push(basis.last().unwrap() * (x - grid.i((.., i - 1))));
    }
    Tensor::stack(&basis, 2)
}

/// Represents a symbolic layer in the Kolmogorov-Arnold Network (KAN).
///
/// The `SymbolicKANLayer` struct contains the necessary parameters and configurations
/// for a symbolic layer in the KAN model.
///
/// # Fields
///
/// * `in_dim` - The input dimension of the layer.
/// * `out_dim` - The output dimension of the layer.
/// * `mask` - The mask of spline functions. Setting an element to zero means setting the corresponding activation to zero function.
/// * `affine` - The affine transformation matrix.
/// * `funs_sympy` - The symbolic representation of the activation functions using SymPy.
struct SymbolicKANLayer {
    in_dim: usize,
    out_dim: usize,
    mask: Tensor,
    affine: Tensor,
    funs_sympy: Vec<Vec<String>>,
}

impl SymbolicKANLayer {
    fn new(in_dim: usize, out_dim: usize, device: Device) -> Self {
        let mask = Tensor::ones(&[out_dim, in_dim], (Kind::Float, device));
        let affine = Tensor::zeros(&[out_dim, in_dim, 4], (Kind::Float, device));
        let funs_sympy = vec![vec!["".to_string(); in_dim]; out_dim];

        SymbolicKANLayer {
            in_dim,
            out_dim,
            mask,
            affine,
            funs_sympy,
        }
    }

    fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let mut x_symbolic = Tensor::zeros(&[x.size()[0], self.out_dim], (Kind::Float, x.device()));
        let mut postacts_symbolic =
            Tensor::zeros(&[x.size()[0], self.out_dim], (Kind::Float, x.device()));

        for j in 0..self.out_dim {
            for i in 0..self.in_dim {
                if self.mask[j][i].double_value(&[]) > 0.0 {
                    let a = self.affine[j][i][0].double_value(&[]);
                    let b = self.affine[j][i][1].double_value(&[]);
                    let c = self.affine[j][i][2].double_value(&[]);
                    let d = self.affine[j][i][3].double_value(&[]);
                    let fun_name = &self.funs_sympy[j][i];

                    let result = match fun_name.as_str() {
                        "sin" => (a * x.index(&[.., i]) + b).sin() * c + d,
                        "cos" => (a * x.index(&[.., i]) + b).cos() * c + d,
                        "tanh" => (a * x.index(&[.., i]) + b).tanh() * c + d,
                        "exp" => (a * x.index(&[.., i]) + b).exp() * c + d,
                        "sig" => (a * x.index(&[.., i]) + b).sigmoid() * c + d,
                        "log" => (a * x.index(&[.., i]) + b + 1e-4).log() * c + d,
                        "sqrt" => (a * x.index(&[.., i]) + b + 1e-4).sqrt() * c + d,
                        "cosh" => (a * x.index(&[.., i]) + b).cosh() * c + d,
                        "sinh" => (a * x.index(&[.., i]) + b).sinh() * c + d,
                        "id" => a * x.index(&[.., i]) + b,
                        _ => Tensor::zeros_like(&x.index(&[.., i])),
                    };

                    x_symbolic.index_mut(&[.., j]).add_(&result);
                    postacts_symbolic.index_mut(&[.., j]).add_(&result);
                }
            }
        }

        (x_symbolic, postacts_symbolic)
    }

    fn fix_symbolic(&mut self, i: usize, j: usize, fun_name: &str, a: f64, b: f64, c: f64, d: f64) {
        if fun_name == "sin"
            || fun_name == "cos"
            || fun_name == "tanh"
            || fun_name == "exp"
            || fun_name == "sig"
            || fun_name == "log"
            || fun_name == "sqrt"
            || fun_name == "cosh"
            || fun_name == "sinh"
            || fun_name == "id"
        {
            self.funs_sympy[j][i] = fun_name.to_string();
            self.affine[j][i][0] = a.into();
            self.affine[j][i][1] = b.into();
            self.affine[j][i][2] = c.into();
            self.affine[j][i][3] = d.into();
        } else {
            panic!("Invalid function name: {}", fun_name);
        }
    }

    fn fix_symbolic_from_data(
        &mut self,
        i: usize,
        j: usize,
        fun_name: &str,
        x: &Tensor,
        y: &Tensor,
    ) -> f64 {
        let (a, b, c, d) = match fun_name {
            "sin" | "cos" | "tanh" | "exp" | "sig" | "log" | "sqrt" | "cosh" | "sinh" => {
                let (a, b, c, d) = self.fit_affine(x, y, fun_name);
                (
                    a.double_value(&[]),
                    b.double_value(&[]),
                    c.double_value(&[]),
                    d.double_value(&[]),
                )
            }
            "id" => {
                let (a, b) = self.fit_linear(x, y);
                (a.double_value(&[]), b.double_value(&[]), 1.0, 0.0)
            }
            _ => panic!("Invalid function name: {}", fun_name),
        };

        self.fix_symbolic(i, j, fun_name, a, b, c, d);

        let y_pred = match fun_name {
            "sin" => (a * x + b).sin() * c + d,
            "cos" => (a * x + b).cos() * c + d,
            "tanh" => (a * x + b).tanh() * c + d,
            "exp" => (a * x + b).exp() * c + d,
            "sig" => (a * x + b).sigmoid() * c + d,
            "log" => (a * x + b + 1e-4).log() * c + d,
            "sqrt" => (a * x + b + 1e-4).sqrt() * c + d,
            "cosh" => (a * x + b).cosh() * c + d,
            "sinh" => (a * x + b).sinh() * c + d,
            "id" => a * x + b,
            _ => panic!("Invalid function name: {}", fun_name),
        };

        let sse = (y - y_pred).pow(2).sum().double_value(&[]);
        let sst = (y - y.mean(&[0], false, Kind::Float))
            .pow(2)
            .sum()
            .double_value(&[]);

        1.0 - sse / sst
    }

    fn fit_affine(
        &self,
        x: &Tensor,
        y: &Tensor,
        fun_name: &str,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        // Implementation of fitting affine parameters for a given function
        // ...
    }

    fn fit_linear(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        // Implementation of fitting linear parameters
        // ...
    }

    fn get_subset(&self, active_in: &[usize], active_out: &[usize]) -> Self {
        let sub_in_dim = active_in.len();
        let sub_out_dim = active_out.len();
        let sub_mask = self.mask.index(&[active_out, active_in]);
        let sub_affine = self.affine.index(&[active_out, active_in, ..]);
        let sub_funs_sympy = active_out
            .iter()
            .map(|&j| {
                active_in
                    .iter()
                    .map(|&i| self.funs_sympy[j][i].clone())
                    .collect()
            })
            .collect();

        Symbolic_KANLayer {
            in_dim: sub_in_dim,
            out_dim: sub_out_dim,
            mask: sub_mask,
            affine: sub_affine,
            funs_sympy: sub_funs_sympy,
        }
    }
}

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

fn main() {
    println!("Hello, world!");
}
