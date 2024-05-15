# `kan-rs`: Kolmogorov-Arnold Networks in 🦀

`kan-rs` is a Rust implementation of [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs), a novel neural network architecture inspired by the Kolmogorov-Arnold representation theorem. KANs offer a unique approach to function approximation by combining continuous piecewise polynomial splines with symbolic functions. Specifically . It is a port of [py-kan](https://github.com/KindXiaoming/pykan/tree/master). 

## Key Innovations

- **Polynomial Spline Activation Functions**: Unlike traditional MLPs that rely on fixed activation functions, KANs utilize continuous piecewise polynomial splines as activation functions. These splines provide a more flexible and expressive representation of complex functions.

- **Symbolic Representation**: KANs introduce a symbolic representation layer alongside the numerical spline-based layer. This symbolic layer allows for interpretable and analytically tractable expressions of the learned function.

- **Function Approximation**: KANs leverage the Kolmogorov-Arnold representation theorem to approximate any continuous multivariate function as a composition of univariate functions and the binary operation of addition. This theoretical foundation enables KANs to effectively learn and represent a wide range of functions.

## Differences between py-KAN and `kan-rs`

- **Sympy**: I have employed a rough hack to emulate Sympy's Computer Algebra System (CAS). I am still evaluating better ways to implement this.

## Limitations


- **Limited Functionality**: `kan-rs` provides a basic implementation of KANs and may not include all the features and extensions available in py-KAN. Additional functionality and customization options could be added to enhance the usability and flexibility of `kan-rs`.

- **Experimental Nature**: KANs are a relatively new architecture, and their effectiveness and limitations are still being explored. While `kan-rs` aims to provide a faithful implementation, further research and empirical evaluation are necessary to fully understand the capabilities and limitations of KANs in various domains.

## Educational Purpose

`kan-rs` is developed for educational purposes, serving as a learning resource for those interested in understanding and implementing KANs in Rust. It provides a starting point for exploring the concepts and techniques behind KANs and encourages experimentation and further development.

However, it is important to note that `kan-rs` is not production-ready (and probably never will be) and should not be used in critical or real-world applications without thorough testing, validation, and additional development efforts. The implementation may have limitations, bugs, or performance issues that need to be addressed before considering it for production use.

## Getting Started

To get started with `kan-rs`, refer to the documentation and examples provided in the repository. The code is organized into modules, and the main entry point is located in the `src/main.rs` file. You can run the code using the Rust compiler and experiment with different configurations and datasets.

## Contributing

Contributions to `kan-rs` are welcome! If you encounter any issues, have suggestions for improvements, or would like to add new features, please feel free to open an issue or submit a pull request. However, keep in mind that this project is primarily for educational purposes, and the focus is on learning and understanding the concepts behind KANs.

## License

`kan-rs` is open-source and released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for educational and non-commercial purposes. However, please note that the authors and contributors of `kan-rs` are not liable for any damages or consequences arising from the use of this software.

## Acknowledgments

`kan-rs` is inspired by the original py-KAN implementation and the research on Kolmogorov-Arnold Representation Networks. We would like to acknowledge the authors and contributors of py-KAN for their valuable work and insights.

If you have any questions or need further assistance, please feel free to reach out. Happy learning and experimentation with `kan-rs`!

## TODOs

- [ ] Add tests
- [ ] Fix local torch installation https://github.com/LaurentMazare/tch-rs/issues/488
- [ ] Find sane way of interfacing with Sympy , or suitable replacement .  Py03?
- [x] Visualization
- [ ] [Rust Jupyter Notebook](https://github.com/evcxr/evcxr)