# Overview
`seqinf` is a Python library for performing sequential simulation-based
inference. It aims to complement the popular [`sbi`][1] package by providing
top-level abstractions that enable running methods like SNPE/SNLE/etc without
additional boilerplate. It also provides additional inference diagnostics and
support for Bayesian NDEs and active methods (e.g., ASNPE).

# Install
Installation is available through PyPI:

```sh
pip install seqinf
```

## Dependencies
Note the version clash between `sbibm`'s requirement for `sbi`, and the latest
`sbi`. This can more or less be ignored, and if this presents an issue when
installing locally, just make sure to install `sbi` _after_ `sbibm`.



# Citing this package
Although `seqinf` is intended for general pipelines, it was principally written
to wrap [ASNPE][2]. If you use this library in your work, please cite the
following:

```
@misc{griesemer2024activesequentialposteriorestimation,
      title={Active Sequential Posterior Estimation for Sample-Efficient Simulation-Based Inference}, 
      author={Sam Griesemer and Defu Cao and Zijun Cui and Carolina Osorio and Yan Liu},
      year={2024},
      eprint={2412.05590},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.05590}, 
}
```


[1]: https://github.com/sbi-dev/sbi
[2]: https://arxiv.org/abs/2412.05590
