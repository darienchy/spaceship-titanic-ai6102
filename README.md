# Spaceship Titanic: AI6102 Group Project

NTU AI6102 Machine Learning Methodologies & Applications, term project on the Kaggle competition
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic): predict which passengers
were transported to another dimension (binary classification, scored by accuracy).

Team: _add names here_

## Results

| Model | CV accuracy (grouped 5-fold, seed 0) | Public leaderboard |
| --- | --- | --- |
| Baseline LightGBM, raw features | 0.796 (3 seeds) | 0.79424 |
| CatBoost, default | 0.8169 | 0.80757 |
| TabPFN | 0.8171 | 0.81295 |
| **Final: 50/50 blend of TabPFN + CatBoost** | **0.8201** | **0.81248** |

## Repository layout

| Path | Contents |
| --- | --- |
| `ss_utils.py` | Shared helpers: data loading, imputation, feature groups, grouped CV, logging, submissions |
| `01_baseline.ipynb` | Baseline model and first submission |
| `02_eda.ipynb` | Exploratory data analysis and report figures |
| `03_features.ipynb` | E1 feature-group ablation, E2 imputation strategies |
| `04_models.ipynb` | E3 validation scheme, E4 model comparison and Optuna tuning, E5 ensembling |
| `05_analysis.ipynb` | E5 follow-up, E6 SHAP, E7 threshold and pseudo-labelling, E8 CV vs leaderboard, error analysis |
| `06_final.ipynb` | Final submission checks and exports |
| `results/` | Experiment log, tuned hyperparameters, out-of-fold predictions, LaTeX tables |
| `figures/` | Figures used in the report |
| `submissions/` | Every file submitted to Kaggle |
| `Screenshots of Leaderboard/` | Leaderboard and submission screenshots |

## Reproducing

1. Create the environment:
   ```bash
   conda create -n ai6102 python=3.11 -y
   conda activate ai6102
   conda install -c conda-forge pandas scikit-learn matplotlib seaborn lightgbm xgboost catboost optuna shap jupyter -y
   ```
2. Download `train.csv`, `test.csv` and `sample_submission.csv` from the
   [competition data page](https://www.kaggle.com/competitions/spaceship-titanic/data) into this folder.
   They are not included here, per Kaggle's competition rules.
3. Run the notebooks in order, 01 to 06. All runs append to `results/experiment_log.csv`.
4. TabPFN needs a GPU. It was run on a free Google Colab T4 with the same features and folds,
   and its predictions are saved in `results/tabpfn_oof.csv` and `results/tabpfn_test_probs.csv`.
