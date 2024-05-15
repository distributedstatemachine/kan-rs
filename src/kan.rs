use crate::kan_layer::KANLayer;
use crate::symbolic_kan_layer::SymbolicKANLayer;
use rand::prelude::*;
use tch::{nn, Device, Kind, Tensor};

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
    device: Device,
}

impl KAN {
    fn new(
        width: &[usize],
        grid: usize,
        k: usize,
        base_fun: nn::Module,
        symbolic_enabled: bool,
        device: Device,
    ) -> Self {
        let depth = width.len() - 1;
        let mut biases = Vec::with_capacity(depth);
        let mut act_fun = Vec::with_capacity(depth);
        let mut symbolic_fun = Vec::with_capacity(depth);

        for l in 0..depth {
            let bias = nn::linear(&vs.root(), width[l], width[l + 1], Default::default());
            biases.push(bias);

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

            let symbolic_layer = Symbolic_KANLayer::new(width[l], width[l + 1], device);
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
                self.act_fun[l].forward(&acts[l]);
            let (x_symbolic, postacts_symbolic) = if self.symbolic_enabled {
                self.symbolic_fun[l].forward(&acts[l])
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

    fn initialize_from_another_model(&mut self, another_model: &KAN, x: &Tensor) {
        let _ = another_model.forward(x);
        let batch = x.size()[0];

        self.initialize_grid_from_another_model(another_model, x);

        for l in 0..self.depth {
            let spb = &mut self.act_fun[l];
            let spb_parent = &another_model.act_fun[l];

            let preacts = &another_model.spline_preacts[l];
            let postsplines = &another_model.spline_postsplines[l];
            spb.coef.copy_(&curve2coef(
                preacts.view([batch, spb.size()]).transpose(0, 1),
                postsplines.view([batch, spb.size()]).transpose(0, 1),
                &spb.grid,
                spb.k,
                self.device,
            ));
            spb.scale_base.copy_(&spb_parent.scale_base);
            spb.scale_sp.copy_(&spb_parent.scale_sp);
            spb.mask.copy_(&spb_parent.mask);
        }

        for l in 0..self.depth {
            self.biases[l].ws.copy_(&another_model.biases[l].ws);
        }

        for l in 0..self.depth {
            self.symbolic_fun[l] = another_model.symbolic_fun[l].clone();
        }
    }

    fn update_grid_from_samples(&mut self, x: &Tensor) {
        for l in 0..self.depth {
            let _ = self.forward(x);
            self.act_fun[l].update_grid_from_samples(&self.acts[l]);
        }
    }

    fn initialize_grid_from_another_model(&mut self, model: &KAN, x: &Tensor) {
        let _ = model.forward(x);
        for l in 0..self.depth {
            self.act_fun[l].initialize_grid_from_parent(&model.act_fun[l], &model.acts[l]);
        }
    }

    fn set_mode(&mut self, l: usize, i: usize, j: usize, mode: &str) {
        match mode {
            "s" => {
                self.act_fun[l].mask[j * self.act_fun[l].in_dim + i] = 0.0;
                self.symbolic_fun[l].mask[j][i] = 1.0;
            }
            "n" => {
                self.act_fun[l].mask[j * self.act_fun[l].in_dim + i] = 1.0;
                self.symbolic_fun[l].mask[j][i] = 0.0;
            }
            "sn" | "ns" => {
                self.act_fun[l].mask[j * self.act_fun[l].in_dim + i] = 1.0;
                self.symbolic_fun[l].mask[j][i] = 1.0;
            }
            _ => {
                self.act_fun[l].mask[j * self.act_fun[l].in_dim + i] = 0.0;
                self.symbolic_fun[l].mask[j][i] = 0.0;
            }
        }
    }

    fn fix_symbolic(
        &mut self,
        l: usize,
        i: usize,
        j: usize,
        fun_name: &str,
        fit_params: bool,
    ) -> f64 {
        self.set_mode(l, i, j, "s");
        if !fit_params {
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, 1.0, 0.0, 1.0, 0.0);
            0.0
        } else {
            let x = &self.acts[l].index(&[.., i]);
            let y = &self.spline_postacts[l].index(&[.., j, i]);
            self.symbolic_fun[l].fix_symbolic_from_data(i, j, fun_name, x, y)
        }
    }

    fn unfix_symbolic(&mut self, l: usize, i: usize, j: usize) {
        self.set_mode(l, i, j, "n");
    }

    fn unfix_symbolic_all(&mut self) {
        for l in 0..self.depth {
            for i in 0..self.width[l] {
                for j in 0..self.width[l + 1] {
                    self.unfix_symbolic(l, i, j);
                }
            }
        }
    }

    fn lock(&mut self, l: usize, ids: &[(usize, usize)]) {
        self.act_fun[l].lock(ids);
    }

    fn unlock(&mut self, l: usize, ids: &[(usize, usize)]) {
        self.act_fun[l].unlock(ids);
    }

    fn get_range(&self, l: usize, i: usize, j: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let x = &self.spline_preacts[l].index(&[.., j, i]);
        let y = &self.spline_postacts[l].index(&[.., j, i]);
        let x_min = x.min().unwrap();
        let x_max = x.max().unwrap();
        let y_min = y.min().unwrap();
        let y_max = y.max().unwrap();
        (x_min, x_max, y_min, y_max)
    }

    fn auto_symbolic(&mut self, lib: &[&str]) {
        for l in 0..self.depth {
            for i in 0..self.width[l] {
                for j in 0..self.width[l + 1] {
                    if self.symbolic_fun[l].mask[j][i].double_value(&[]) > 0.0 {
                        println!("skipping ({}, {}, {}) since already symbolic", l, i, j);
                    } else {
                        let (name, _, r2) = self.suggest_symbolic(l, i, j, lib);
                        self.fix_symbolic(l, i, j, &name, true);
                        println!("fixing ({}, {}, {}) with {}, r2={}", l, i, j, name, r2);
                    }
                }
            }
        }
    }

    fn suggest_symbolic(
        &mut self,
        l: usize,
        i: usize,
        j: usize,
        lib: &[&str],
    ) -> (String, String, f64) {
        let mut r2s = Vec::new();

        for &fun_name in lib {
            let r2 = self.fix_symbolic(l, i, j, fun_name, true);
            r2s.push(r2);
        }

        self.unfix_symbolic(l, i, j);

        let best_index = r2s
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let best_name = lib[best_index].to_string();
        let best_r2 = r2s[best_index];

        (best_name, best_name, best_r2)
    }

    fn prune(&mut self, threshold: f64) -> Self {
        let mut mask = vec![Tensor::ones(&[self.width[0]], (Kind::Float, self.device))];
        let mut active_neurons = vec![Vec::from_iter(0..self.width[0])];

        for i in 0..self.depth - 1 {
            let in_important = self.acts_scale[i].max_dim(1, false).0.gt(threshold);
            let out_important = self.acts_scale[i + 1].max_dim(0, false).0.gt(threshold);
            let overall_important = &in_important & &out_important;

            mask.push(overall_important.to_kind(Kind::Float));
            active_neurons.push(overall_important.nonzero().view([-1]).to_vec());
        }

        active_neurons.push(Vec::from_iter(0..self.width[self.depth]));
        mask.push(Tensor::ones(
            &[self.width[self.depth]],
            (Kind::Float, self.device),
        ));

        self.mask = mask;

        for l in 0..self.depth - 1 {
            for i in 0..self.width[l + 1] {
                if !active_neurons[l + 1].contains(&i) {
                    self.remove_node(l + 1, i);
                }
            }
        }

        let mut model2 = KAN::new(
            &self.width,
            self.grid,
            self.k,
            self.base_fun.clone(),
            self.symbolic_enabled,
            self.device,
        );
        model2.load_state_dict(&self.state_dict());

        for i in 0..self.depth {
            if i < self.depth - 1 {
                model2.biases[i].ws.copy_(&self.biases[i].ws.index([
                    Tensor::arange(active_neurons[i + 1].len(), (Kind::Int64, self.device)),
                    Tensor::of_slice(&active_neurons[i + 1]),
                ]));
            }

            model2.act_fun[i] =
                self.act_fun[i].get_subset(&active_neurons[i], &active_neurons[i + 1]);
            model2.width[i] = active_neurons[i].len();
            model2.symbolic_fun[i] =
                self.symbolic_fun[i].get_subset(&active_neurons[i], &active_neurons[i + 1]);
        }

        model2
    }

    fn remove_edge(&mut self, l: usize, i: usize, j: usize) {
        self.act_fun[l].mask[j * self.width[l] + i] = 0.0;
    }

    fn remove_node(&mut self, l: usize, i: usize) {
        self.act_fun[l - 1]
            .mask
            .index_mut(&[i * self.width[l - 1]
                + Tensor::arange(self.width[l - 1], (Kind::Int64, self.device))])
            .fill_(0.0);
        self.act_fun[l]
            .mask
            .index_mut(&[
                Tensor::arange(self.width[l + 1], (Kind::Int64, self.device)) * self.width[l] + i,
            ])
            .fill_(0.0);
        self.symbolic_fun[l - 1].mask.index_mut(&[i, ..]).fill_(0.0);
        self.symbolic_fun[l].mask.index_mut(&[.., i]).fill_(0.0);
    }
}

fn curve2coef(
    preacts: Tensor,
    postacts: Tensor,
    grid: &Tensor,
    k: usize,
    device: Device,
) -> Tensor {
    let batch_size = preacts.size()[1];
    let num_grid = grid.size()[0] - 1;

    let mut coefficients = Vec::with_capacity(k + 1);
    for _ in 0..=k {
        coefficients.push(Vec::with_capacity(num_grid));
    }

    for i in 0..num_grid {
        let mask = (&preacts >= grid[i]) & (&preacts < grid[i + 1]);
        let x = preacts.where1(
            &mask,
            &Tensor::full(&[1], f64::NEG_INFINITY, (Kind::Float, device)),
        );
        let y = postacts.where1(
            &mask,
            &Tensor::full(&[1], f64::NEG_INFINITY, (Kind::Float, device)),
        );

        if x.size()[0] > 0 {
            let basis = polynomial_basis(
                &x.unsqueeze(-1),
                &grid.slice(0, i, i + 2, 1).unsqueeze(1),
                k,
            );
            // TODO: placeholder , find more suitable value for rcond.
            let coef = basis.pinverse(1e-15).matmul(&y.unsqueeze(-1)).squeeze();

            for j in 0..=k {
                coefficients[j].push(coef.index([j]));
            }
        } else {
            for j in 0..=k {
                coefficients[j].push(Tensor::zeros(&[], (Kind::Float, device)));
            }
        }
    }

    let coefficients = coefficients
        .into_iter()
        .map(|coef| Tensor::stack(&coef, 0))
        .collect::<Vec<_>>();
    Tensor::stack(&coefficients, 1).view([batch_size, -1, k + 1])
}

fn polynomial_basis(x: &Tensor, grid: &Tensor, k: usize) -> Tensor {
    let mut basis = vec![Tensor::ones_like(x)];
    for i in 1..=k {
        basis.push(basis.last().unwrap() * (x - grid.index(0)));
    }
    Tensor::stack(&basis, -1)
}

trait StateDict {
    fn state_dict(&self) -> Vec<(&str, Tensor)>;
    fn load_state_dict(&mut self, state_dict: &[(&str, Tensor)]);
}

impl StateDict for KAN {
    fn state_dict(&self) -> Vec<(&str, Tensor)> {
        let mut state_dict = Vec::new();

        for (i, bias) in self.biases.iter().enumerate() {
            state_dict.push((format!("biases.{}.weight", i).as_str(), bias.ws.copy()));
        }

        for (i, act_fun) in self.act_fun.iter().enumerate() {
            state_dict.push((
                format!("act_fun.{}.scale_base", i).as_str(),
                act_fun.scale_base.clone(),
            ));
            state_dict.push((
                format!("act_fun.{}.scale_sp", i).as_str(),
                act_fun.scale_sp.clone(),
            ));
            state_dict.push((format!("act_fun.{}.coef", i).as_str(), act_fun.coef.clone()));
            state_dict.push((format!("act_fun.{}.mask", i).as_str(), act_fun.mask.clone()));
        }

        for (i, symbolic_fun) in self.symbolic_fun.iter().enumerate() {
            state_dict.push((
                format!("symbolic_fun.{}.mask", i).as_str(),
                symbolic_fun.mask.clone(),
            ));
            state_dict.push((
                format!("symbolic_fun.{}.affine", i).as_str(),
                symbolic_fun.affine.clone(),
            ));
        }

        state_dict
    }

    fn load_state_dict(&mut self, state_dict: &[(&str, Tensor)]) {
        for (name, tensor) in state_dict {
            if name.starts_with("biases") {
                let index = name.split('.').nth(1).unwrap().parse::<usize>().unwrap();
                self.biases[index].ws.copy_(tensor);
            } else if name.starts_with("act_fun") {
                let index = name.split('.').nth(1).unwrap().parse::<usize>().unwrap();
                match name.split('.').nth(2).unwrap() {
                    "scale_base" => self.act_fun[index].scale_base.copy_(tensor),
                    "scale_sp" => self.act_fun[index].scale_sp.copy_(tensor),
                    "coef" => self.act_fun[index].coef.copy_(tensor),
                    "mask" => self.act_fun[index].mask.copy_(tensor),
                    _ => panic!("Invalid state dict key: {}", name),
                }
            } else if name.starts_with("symbolic_fun") {
                let index = name.split('.').nth(1).unwrap().parse::<usize>().unwrap();
                match name.split('.').nth(2).unwrap() {
                    "mask" => self.symbolic_fun[index].mask.copy_(tensor),
                    "affine" => self.symbolic_fun[index].affine.copy_(tensor),
                    _ => panic!("Invalid state dict key: {}", name),
                }
            } else {
                panic!("Invalid state dict key: {}", name);
            }
        }
    }
}
