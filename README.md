# [F]()ormula graph self-attent[i]()on [n]()etwork for representation-domain independent materials [d]()iscov[er]()y ([Finder](https://onlinelibrary.wiley.com/doi/10.1002/advs.202200164))

<p align="center">
<img src="https://github.com/ihalage/Finder/blob/main/Finder/figures/figure2.png" align="center" width="600">
</p>

## Installation

This project requires `python3.8` or above. Please make sure you have the `pip3` module installed. It is recommended to use a virtual environment to install the `Finder` package as follows.

```
python -m venv Finder_env
source Finder_env/bin/activate
```

Inside the root directory, execute `pip install -r requirements.txt` to install the dependencies. This should install all packages required to run `Finder`. Please open an issue if there are installation errors.

`Finder` is built using [spektral](https://graphneural.network/) graph deep learning library. You may read the documentation of spektral [here](https://graphneural.network/getting-started/). Note that the current version of `Finder` requires `spektral-1.1.0`.

## Database

Please download the The Materials Project data used in this work from [figshare](https://doi.org/10.6084/m9.figshare.19308407). Extract the zip file and place `MP_2021_July_no_polymorph` directory inside `data/databases/`. Note that each data file should have three columns `ID`, `formula` and `target`. An additional `cif` column is required for crystal structure based predictions.

## Usage

Navigate to the main directory (`Finder/`) and execute `python trainer.py --help` to see the allowed arguments.

### Structure-agnostic Finder

You can train and evaluate structure-agnostic `Finder` model on the formation energy database by running the following.

```
python trainer.py --train-path data/databases/MP_2021_July_no_polymorph/formation_energy/train.csv --val-path data/databases/MP_2021_July_no_polymorph/formation_energy/val.csv --test-path data/databases/MP_2021_July_no_polymorph/formation_energy/test.csv --epochs 800 --batch-size 128 --train --test
```

### Structure-based Finder

An additional `--use-crystal-structure` flag is required to train structure-based `Finder` model. To train it for bandgap, you can run;

```
python trainer.py --train-path data/databases/MP_2021_July_no_polymorph/bandgap/train.csv --val-path data/databases/MP_2021_July_no_polymorph/bandgap/val.csv --test-path data/databases/MP_2021_July_no_polymorph/bandgap/test.csv --epochs 1200 --batch-size 128 --train --test --use-crystal-structure
```

### How to predict materials properties using a pre-trained Finder model?

Once you train a `Finder` model, a directory named `saved_models/best_model_gnn` that contains the best model will be created. You may then run the following to make predictions using this pre-trained model. Add `--use-crystal-structure` flag if this is a structure-based `Finder` model. If the target property value is unknown, please fill in the `target` column of your data file with some dummy values.

```
python trainer.py --model-path saved_models/best_model_gnn/ --test-path data/databases/MP_2021_July_no_polymorph/formation_energy/test.csv --test
```

Prediction results will be saved in `results/` directory. 

You may download the pre-trained `Finder` models for MP property prediction tasks from [figshare](https://doi.org/10.6084/m9.figshare.19308392). Assuming that the zip file is extracted in the root directory, you may run the following snippet to evaluate,  for example the structure-based refractive index model.

```
python trainer.py --model-path Finder_pre-trained/Structure-based/best_model_gnn_refractive_index_SB/ --test-path data/databases/MP_2021_July_no_polymorph/refractive_index/test.csv --test --use-crystal-structure
```


## Funding
We acknowledge funding received by The Institution of Engineering and Technology (IET) under the AF Harvey Research Prize. This work is supported in part by EPSRC Software Defined Materials for Dynamic Control of Electromagnetic Waves (ANIMATE) grant (No. EP/R035393/1) 

## Citation
Consider citing our paper if you find the `Finder` model and the codebase useful.

```
@article{Ihalage_2022_Adv_Sci,
author = {Ihalage, Achintha and Hao, Yang},
title = {Formula Graph Self-Attention Network for Representation-Domain Independent Materials Discovery},
journal = {Advanced Science},
volume = {9},
number = {18},
pages = {2200164},
keywords = {attention, epsilon-near-zero, graph-network, machine-learning, materials-informatics},
doi = {https://doi.org/10.1002/advs.202200164},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202200164},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202200164},
year = {2022}
}
```
