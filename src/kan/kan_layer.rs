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
pub struct KANLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num: usize,
    pub k: usize,
    pub scale_base: Tensor,
    pub scale_sp: Tensor,
    pub base_fun: nn::Module,
    pub coef: Tensor,
    pub mask: Tensor,
    pub grid: Tensor,
    pub sp_trainable: bool,
    pub sb_trainable: bool,
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

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
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

    pub fn get_subset(&self, active_in: &[usize], active_out: &[usize]) -> Self {
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

    pub fn lock(&mut self, ids: &[(usize, usize)]) {
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

    pub fn initialize_grid_from_parent(&mut self, parent: &KANLayer, activations: &Tensor) {
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

    pub fn update_grid_from_samples(&mut self, samples: &Tensor) {
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
