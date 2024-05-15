use tch::{nn, Device, Kind, Tensor};

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
pub struct SymbolicKANLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub mask: Tensor,
    pub affine: Tensor,
    pub funs_sympy: Vec<Vec<String>>,
}

impl SymbolicKANLayer {
    pub fn new(in_dim: usize, out_dim: usize, device: Device) -> Self {
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

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
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
                        "sin" => (a * x.index((.., i)) + b).sin() * c + d,
                        "cos" => (a * x.index((.., i)) + b).cos() * c + d,
                        "tanh" => (a * x.index((.., i)) + b).tanh() * c + d,
                        "exp" => (a * x.index((.., i)) + b).exp() * c + d,
                        "sig" => (a * x.index((.., i)) + b).sigmoid() * c + d,
                        "log" => (a * x.index((.., i)) + b + 1e-4).log() * c + d,
                        "sqrt" => (a * x.index((.., i)) + b + 1e-4).sqrt() * c + d,
                        "cosh" => (a * x.index((.., i)) + b).cosh() * c + d,
                        "sinh" => (a * x.index((.., i)) + b).sinh() * c + d,
                        "id" => a * x.index((.., i)) + b,
                        _ => Tensor::zeros_like(&x.index((.., i))),
                    };

                    x_symbolic.index_mut(&[.., j]).add_(&result);
                    postacts_symbolic.index_mut(&[.., j]).add_(&result);
                }
            }
        }

        (x_symbolic, postacts_symbolic)
    }

    pub fn fix_symbolic(&mut self, i: usize, j: usize, fun_name: &str, a: f64, b: f64, c: f64, d: f64) {
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

    pub fn fix_symbolic_from_data(
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

        let sse = (y - y_pred).pow(2).sum_dim(&[0], false).double_value(&[]);
    let sst = (y - y.mean(Kind::Float)).pow(2).sum_dim(&[0], false).double_value(&[]);

        1.0 - sse / sst
    }

    fn fit_affine(
        &self,
        x: &Tensor,
        y: &Tensor,
        fun_name: &str,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        match fun_name {
            "sin" | "cos" | "tanh" | "exp" | "sig" | "log" | "sqrt" | "cosh" | "sinh" => {
                let (a, b) = self.fit_linear(x, y);
                let y_pred = match fun_name {
                    "sin" => (a * x + b).sin(),
                    "cos" => (a * x + b).cos(),
                    "tanh" => (a * x + b).tanh(),
                    "exp" => (a * x + b).exp(),
                    "sig" => (a * x + b).sigmoid(),
                    "log" => (a * x + b + 1e-4).log(),
                    "sqrt" => (a * x + b + 1e-4).sqrt(),
                    "cosh" => (a * x + b).cosh(),
                    "sinh" => (a * x + b).sinh(),
                    _ => unreachable!(),
                };
                let c = (y - y_pred).mean(Kind::Float);
                let d = y_pred.mean(Kind::Float);
                (a, b, c, d)
            }
            "id" => {
                let (a, b) = self.fit_linear(x, y);
                (
                    a,
                    b,
                    Tensor::ones(&[], (Kind::Float, x.device())),
                    Tensor::zeros(&[], (Kind::Float, x.device())),
                )
            }
            _ => panic!("Invalid function name: {}", fun_name),
        }
    }

    fn fit_linear(&self, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let x_mean = x.mean(Kind::Float);
        let y_mean = y.mean(Kind::Float);
        let x_centered = x - x_mean;
        let y_centered = y - y_mean;
        let a = x_centered.dot(&y_centered) / x_centered.pow(2).sum(Kind::Float);
        let b = y_mean - a * x_mean;
        (a, b)
    }

    pub fn get_subset(&self, active_in: &[usize], active_out: &[usize]) -> Self {
        let sub_in_dim = active_in.len();
        let sub_out_dim = active_out.len();
        let active_out_tensor = Tensor::of_slice(active_out).to_device(self.mask.device());
        let active_in_tensor = Tensor::of_slice(active_in).to_device(self.mask.device());
        let sub_mask = self.mask.index_select(0, &active_out_tensor).index_select(1, &active_in_tensor);
        let sub_affine = self.affine.index_select(0, &active_out_tensor).index_select(1, &active_in_tensor);
        let sub_funs_sympy = active_out
            .iter()
            .map(|&j| {
                active_in
                    .iter()
                    .map(|&i| self.funs_sympy[j][i].clone())
                    .collect()
            })
            .collect();
    
        SymbolicKANLayer {
            in_dim: sub_in_dim,
            out_dim: sub_out_dim,
            mask: sub_mask,
            affine: sub_affine,
            funs_sympy: sub_funs_sympy,
        }
    }
}
