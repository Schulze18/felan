# FeLaN
Open-source code of **Floating-Base Deep Lagrangian Networks (FeLaN)**.

## Overview
FeLaN is a grey-box method for physically consistent system identification (SysID) of floating-base robots (e.g., humanoids, quadrupeds). It learns system dynamics under Lagrangian mechanics while enforcing key structural constraints of floating-base systems (e.g., branch-induced sparsity/decoupling and full physical consistency of the composite spatial inertia) through a novel inertia matrix parametrization.

For a quadruped, FeLaN uses one MLP per leg (four in total), each taking only that leg’s joint positions as input, plus an additional MLP that models the composite spatial inertia terms of the floating base.
<p align="center">
  <img src="assets/felan_architecture.png" width="500">
</p>

<div align="center">
  <a href="#Setup"><b>Setup</b></a> |
  <a href="https://schulze18.github.io/felan_website/"><b>Project Page</b></a> |
  <a href="https://arxiv.org/abs/2510.17270"><b>Preprint</b></a>
</div>

## What’s in this repository

- **Training & evaluation pipelines in JAX** for quadrupeds and humanoids.

- **FeLaN models** (including FeLaNBS) and **reference baselines** (DeLaN/DeLaNPP, MLP).

- **MjxDNEA**: white-box SysID using Mujoxo XLA (MJX) as a differentiable Recursive Newton–Euler algorithm (DNEA). We provide multiple inertial parametrizations.

- **Open-source datasets**: 
   - **Simulated in MJX**: Go2 and Talos
   - **Real**: Spot, Spot with Arm, HyQReal2, and Talos 

See the paper and project page for the method description and experimental results.


## Setup
1. Clone the Repository
   ```bash
   git clone git@github.com:Schulze18/felan.git
   ```

2. Set Up Conda Environment
   ```bash
   conda env create -f felan_env.yml
   conda activate felan
   ```

3. Install FeLaN as a Python pkg
   ```bash
   pip install -e .
   ```

## Datasets

The datasets and robot models used are hosted on Hugging Face.

```bash
git clone https://huggingface.co/datasets/schulze18/felan felan/data/
```


## Quickstart

### Training (quadrupeds)
Available quadruped robots:
- `go2`, `spot_real`, `spot_arm_real`, `hyqreal2`

Supported models (nn):
- `FeLaN`, `FeLaNBS`, `DeLaN`, `DeLaNPP`, `MLP`

Example:
```bash
python -m felan.train_quad --robot go2 --nn FeLaN
```

Train a baseline instead:
```bash
python -m felan.train_quad --robot go2 --nn DeLaN
python -m felan.train_quad --robot go2 --nn MLP
```

If using `MjxDNEA`, you can choose the inertial parametrization:
```bash
python -m felan.train_quad --robot go2 --nn MjxDNEA --inertia_param SpatialLogCholesky
```

Supported inertia parametrization:
- `PrincipalTriangular`, `PrincipalUnconstrained`, `SpatialCov`, `SpatialSpd`, `SpatialLogCholesky`

### Training (humanoids)
Available humanoid robots:
- `talos`, `talos_real`

Example:
```bash
python -m felan.train_humanoid --robot talos --nn FeLaN
```

## Evaluation

To skip trainning and evaluate an existing model, use `-l 1`. For example:
```bash
python -m felan.train_quad --robot go2 --nn DeLaN -l 1
python -m felan.train_humanoid --robot talos --nn FeLaN -l 1
```

Alternatively, you can evaluate a specific saved model directly with:
```bash
python -m felan.evaluate_model \
  --robot talos \
  --model_folder trained_models \
  --nn FeLaN \
  --model_name epochs_3000_talos_sim_freq_100hz_0
```


# ICRA 2026
For the results reported in the ICRA 2026 paper, use the `icra_2026` branch. It includes trained models and the corresponding hyperparameters.

# Acknowledgement
We gratefully acknowledge the support of NVIDIA Corporation through the NVIDIA Academic Grant Program, including the provision of RTX 6000 Ada Generation GPUs used in this research.

# Citing
If you find our work or the provided datasets useful, please consider citing:
```
@misc{schulze2025_felan,
      title={Floating-Base Deep Lagrangian Networks}, 
      author={Lucas Schulze and Juliano Decico Negri and Victor Barasuol and Vivian Suzano Medeiros and Marcelo Becker and Jan Peters and Oleg Arenz},
      year={2025},
      eprint={2510.17270},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.17270}, 
}
```
