<div align="center">
  <a href="https://ccsd.ucsd.edu/home">
    <img align="left" src="figures/CCSD.png" width="400" height="60" alt="ccsd">
  </a>
  <a href="https://ucsd.edu/">
    <img align="right" src="figures/ucsd(1).png" width="260" alt="ucsd">
  </a>
</div>

<br> <br>

# Adaptive Neural-Operator Control for Age-Structured Two-Species Dynamics

<div align="center">
 <a href="#"><img alt="Operator figure" src="figures/tac_operator_figure.png" width="100%"/></a>
</div>

<div align="center">
 <a href="#"><img alt="Adaptive control figure top" src="figures/tac_adaptive_figure_top.png" width="100%"/></a>
</div>

<div align="center">
 <a href="#"><img alt="Adaptive control figure bottom" src="figures/tac_adaptive_figure_bottom.png" width="100%"/></a>
</div>

## About this repository
This repository contains the implementation used to study adaptive control for an age-structured two-species system with neural-operator-based prediction of the Lotka-Sharpe growth rate. The codebase includes:

- Operator construction and diagnostics for the structured population model.
- Simulation scripts for nominal and adaptive TAC experiments.
- Dataset generation pipelines for both the base operator-learning task and the adaptive-estimator task.
- Pretrained FNO checkpoints and the datasets used to train them.

The main figures highlighted above are produced from:
- `scripts/make_tac_operator_figure.py`
- `scripts/make_tac_adaptive.py`

## Examples
To reproduce the main figures from the repository, run:

```bash
python scripts/make_tac_operator_figure.py
python scripts/make_tac_adaptive.py
```

Additional useful scripts:
- `python scripts/make_tac_sim_figure.py` for the nominal simulation figure.
- `python scripts/run_sim.py` for a general simulation rollout with diagnostics.
- `python scripts/gen_adaptive_dataset.py` to regenerate the adaptive estimator dataset.
- `python scripts/train_fno.py` to train the baseline FNO predictor.
- `python scripts/train_fno_adaptive.py` to train the adaptive-estimator FNO predictor.

The repository already includes:
- Datasets in `datasets/`
- Pretrained model checkpoints in `models/`
- Generated figures in `figures/`

## Requirements
The scripts in this repository use the following Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `neuraloperator`

Depending on your environment, you may also want a recent version of `pip` and a CUDA-enabled PyTorch install for faster FNO training.

## Questions or issues
If you have questions or run into issues, please open a GitHub issue for the repository.

## Licensing
This repository does not currently include a top-level license file. Add an explicit license before redistributing or reusing the code outside its intended setting.
