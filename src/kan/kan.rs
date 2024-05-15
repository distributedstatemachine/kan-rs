use crate::kan::kan_layer::KANLayer;
use crate::kan::symbolic_kan_layer::SymbolicKANLayer;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;
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
pub struct KAN {
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
    pub fn new(
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

    pub fn forward(&self, x: &Tensor) -> Tensor {
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

    pub fn train(
        &mut self,
        dataset: &std::collections::HashMap<String, Tensor>,
        opt: &str,
        steps: i64,
        log: i64,
        lamb: f64,
        lamb_l1: f64,
        lamb_entropy: f64,
        lamb_coef: f64,
        lamb_coefdiff: f64,
        update_grid: bool,
        grid_update_num: i64,
        loss_fn: Option<&dyn Fn(&Tensor, &Tensor) -> Tensor>,
        lr: f64,
        stop_grid_update_step: i64,
        batch: i64,
        small_mag_threshold: f64,
        small_reg_factor: f64,
        metrics: &[&dyn Fn() -> f64],
        sglr_avoid: bool,
        save_fig: bool,
        in_vars: &[String],
        out_vars: &[String],
        beta: f64,
        save_fig_freq: i64,
        img_folder: &str,
        device: Device,
    ) -> std::collections::HashMap<String, Vec<f64>> {
        let mut results = std::collections::HashMap::new();
        results.insert("train_loss".to_string(), Vec::new());
        results.insert("test_loss".to_string(), Vec::new());
        results.insert("reg".to_string(), Vec::new());
        for metric in metrics {
            results.insert(
                std::any::type_name::<dyn Fn() -> f64>().to_string(),
                Vec::new(),
            );
        }

        let loss_fn = loss_fn
            .unwrap_or_else(|| &|x: &Tensor, y: &Tensor| (x - y).pow(2).mean(tch::Kind::Float));
        let loss_fn_eval = loss_fn;

        let grid_update_freq = stop_grid_update_step / grid_update_num;

        let optimizer = if opt == "Adam" {
            nn::Adam::default().build(&self.parameters(), lr).unwrap()
        } else {
            nn::Optimizer::lbfgs(&self.parameters(), lr, Default::default())
        };

        let batch_size = if batch == -1 || batch > dataset["train_input"].size()[0] {
            dataset["train_input"].size()[0]
        } else {
            batch
        };
        let batch_size_test = if batch == -1 || batch > dataset["test_input"].size()[0] {
            dataset["test_input"].size()[0]
        } else {
            batch
        };

        let closure = || {
            optimizer.zero_grad();
            let train_id = Tensor::randint(
                dataset["train_input"].size()[0],
                &[batch_size],
                tch::Kind::Int64,
            )
            .to_device(device);
            let pred = self.forward(&dataset["train_input"].index_select(0, &train_id));
            let train_loss = if sglr_avoid {
                let id = tch::no_grad(|| {
                    pred.sum_dim_intlist(&[-1], false, tch::Kind::Float)
                        .isnan()
                        .logical_not()
                });
                loss_fn(
                    &pred.index_select(0, &id),
                    &dataset["train_label"]
                        .index_select(0, &train_id)
                        .index_select(0, &id),
                )
            } else {
                loss_fn(&pred, &dataset["train_label"].index_select(0, &train_id))
            };
            let reg = self.regularization(
                &self.acts_scale,
                lamb_l1,
                lamb_entropy,
                lamb_coef,
                lamb_coefdiff,
                small_mag_threshold,
                small_reg_factor,
            );
            let objective = train_loss + lamb * reg;
            objective.backward();
            train_loss
        };

        for step in 0..steps {
            if step % grid_update_freq == 0 && step < stop_grid_update_step && update_grid {
                self.update_grid_from_samples(&dataset["train_input"].to_device(device));
            }

            let train_loss = if opt == "LBFGS" {
                optimizer.step(closure).unwrap()
            } else {
                closure();
                optimizer.step();
                closure()
            };

            let test_id = Tensor::randint(
                dataset["test_input"].size()[0],
                &[batch_size_test],
                tch::Kind::Int64,
            )
            .to_device(device);
            let test_loss = loss_fn_eval(
                &self.forward(&dataset["test_input"].index_select(0, &test_id)),
                &dataset["test_label"].index_select(0, &test_id),
            );

            if step % log == 0 {
                println!(
                    "Step: {}, Train Loss: {:.4}, Test Loss: {:.4}",
                    step,
                    train_loss.double_value(&[]),
                    test_loss.double_value(&[])
                );
            }

            results
                .get_mut("train_loss")
                .unwrap()
                .push(train_loss.sqrt().double_value(&[]) as f64);
            results
                .get_mut("test_loss")
                .unwrap()
                .push(test_loss.sqrt().double_value(&[]) as f64);
            results.get_mut("reg").unwrap().push(
                lamb * self
                    .regularization(
                        &self.acts_scale,
                        lamb_l1,
                        lamb_entropy,
                        lamb_coef,
                        lamb_coefdiff,
                        small_mag_threshold,
                        small_reg_factor,
                    )
                    .double_value(&[]) as f64,
            );

            for metric in metrics {
                results
                    .get_mut(&format!(
                        "metric_{}",
                        metrics
                            .iter()
                            .position(|&m| std::ptr::eq(m, metric))
                            .unwrap()
                    ))
                    .unwrap()
                    .push(metric());
            }

            if save_fig && step % save_fig_freq == 0 {
                self.plot(img_folder, beta);
            }
        }

        results
    }

    pub fn fix_symbolic(
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

    pub fn auto_symbolic(
        &mut self,
        input_range: (f64, f64),
        output_range: (f64, f64),
        lib: &[&str],
        num_samples: usize,
    ) {
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

    pub fn prune(&mut self, threshold: f64) -> Self {
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

    pub fn visualize(&self, path: &str) {
        let mut views = Vec::new();

        let mut layer_pos = Vec::new();
        let num_layers = self.depth + 1;
        for l in 0..num_layers {
            let y = (l as f64) / ((num_layers - 1) as f64);
            for i in 0..self.width[l] {
                let x = (i as f64 + 0.5) / (self.width[l] as f64);
                layer_pos.push((x, y));
            }
        }

        let mut lines = Vec::new();
        for l in 0..self.depth {
            for i in 0..self.width[l] {
                for j in 0..self.width[l + 1] {
                    let x1 = layer_pos[self.width[0..l].iter().sum::<usize>() + i].0;
                    let y1 = layer_pos[self.width[0..l].iter().sum::<usize>() + i].1;
                    let x2 = layer_pos[self.width[0..=l].iter().sum::<usize>() + j].0;
                    let y2 = layer_pos[self.width[0..=l].iter().sum::<usize>() + j].1;

                    let mask_value = self.act_fun[l].mask[j * self.width[l] + i].double_value(&[]);
                    if mask_value > 0.0 {
                        let color = if self.symbolic_fun[l].mask[j][i].double_value(&[]) > 0.0 {
                            "red"
                        } else {
                            "blue"
                        };
                        lines.push(plotlib::repr::Line::new(vec![(x1, y1), (x2, y2)], color));
                    }
                }
            }
        }
        views.push(Plot::new(lines).point_style(PointStyle::new().colour("black")));

        let mut points = Vec::new();
        for (i, pos) in layer_pos.iter().enumerate() {
            let l = (0..num_layers)
                .find(|&l| {
                    self.width[0..l].iter().sum::<usize>() <= i
                        && i < self.width[0..=l].iter().sum::<usize>()
                })
                .unwrap();
            let color = if l == 0 || l == num_layers - 1 {
                "black"
            } else {
                "blue"
            };
            points.push(plotlib::repr::Point {
                x: pos.0,
                y: pos.1,
                style: PointStyle::new().colour(color).marker(PointMarker::Circle),
            });
        }
        views.push(Plot::new(points));

        let mut page = Page::new(path);
        page.add(ContinuousView {
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            views: views,
        });
        page.render();
    }

    pub fn create_dataset(
        f: &dyn Fn(&Tensor) -> Tensor,
        n_var: i64,
    ) -> std::collections::HashMap<String, Tensor> {
        let train_size = 1000;
        let test_size = 200;
        let bound = 5.0;

        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-bound, bound);

        let train_input = Tensor::of_slice(
            &(0..train_size * n_var)
                .map(|_| dist.sample(&mut rng) as f64)
                .collect::<Vec<f64>>(),
        )
        .reshape(&[train_size, n_var]);

        let test_input = Tensor::of_slice(
            &(0..test_size * n_var)
                .map(|_| dist.sample(&mut rng) as f64)
                .collect::<Vec<f64>>(),
        )
        .reshape(&[test_size, n_var]);

        let train_label = f(&train_input);
        let test_label = f(&test_input);

        let mut dataset = std::collections::HashMap::new();
        dataset.insert("train_input".to_string(), train_input);
        dataset.insert("train_label".to_string(), train_label);
        dataset.insert("test_input".to_string(), test_input);
        dataset.insert("test_label".to_string(), test_label);

        dataset
    }

    pub fn plot(&self, folder: &str, beta: f64) {
        // Create the output folder if it doesn't exist
        std::fs::create_dir_all(folder).expect("Failed to create output folder");

        let depth = self.depth;
        for l in 0..depth {
            for i in 0..self.width[l] {
                for j in 0..self.width[l + 1] {
                    let rank = self.acts[l].slice(0, i, i + 1).argsort(0, true);
                    let mut plot = Plot::new();

                    let symbol_mask = self.symbolic_fun[l].mask.i((j, i));
                    let numerical_mask = self.act_fun[l]
                        .mask
                        .index(&Tensor::of_slice(&[j * self.width[l] + i]));
                    let (color, alpha_mask) = if symbol_mask > 0.0 && numerical_mask > 0.0 {
                        ("purple", 1.0)
                    } else if symbol_mask > 0.0 && numerical_mask == 0.0 {
                        ("red", 1.0)
                    } else if symbol_mask == 0.0 && numerical_mask > 0.0 {
                        ("black", 1.0)
                    } else {
                        ("white", 0.0)
                    };

                    let x: Vec<f64> = rank
                        .iter()
                        .map(|&x| x as f64 / rank.size()[0] as f64)
                        .collect();
                    let y: Vec<f64> = self.acts[l].slice(0, i, i + 1).iter().map(|&y| y).collect();

                    plot.add_trace(
                        Scatter::new(x, y)
                            .mode(plotpy::Mode::Lines)
                            .line(plotpy::Line::new().color(color).width(3.0)),
                    );

                    plot.write_html(format!("{}/sp_{}_{}_{}html", folder, l, i, j));
                }
            }
        }
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

pub trait StateDict {
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
                act_fun.scale_base.copy(),
            ));
            state_dict.push((
                format!("act_fun.{}.scale_sp", i).as_str(),
                act_fun.scale_sp.copy(),
            ));
            state_dict.push((format!("act_fun.{}.coef", i).as_str(), act_fun.coef.copy()));
            state_dict.push((format!("act_fun.{}.mask", i).as_str(), act_fun.mask.copy()));
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
