# [F]()ormula graph self-attent[i]()on [n]()etwork for representation-domain independent materials [d]()iscov[er]()y ([Finder](https://arxiv.org/abs/2201.05649))

## Installation

This project requires `python3.6` or above. Please make sure you have the `pip3` module installed. Using a virtual environment is recommended. Inside the root directory, execute `pip install --ignore-installed -r requirements.txt` to install the dependencies. This should install all packages required to run `Finder`. Please open an issue if there are installation errors.

`Finder` is built using [spektral](https://graphneural.network/) graph deep learning library. You may read the documentation of spektral [here](https://graphneural.network/getting-started/).

## Database

Please download the The Materials Project data used in this work from [figshare](https://figshare.com/articles/dataset/MP_data/19074599). Extract the zip file and place `MP` directory inside `data/databases/`. Note that each data file should have three columns `ID`, `formula` and `target`. An additional `cif` column is required for crystal structure based predictions.

## Usage

Navigate to the main directory (`Finder/`) and execute `python trainer.py --help` to see the allowed arguments.

### Structure-agnostic Finder

You can train and evaluate structure-agnostic `Finder` model on the formation energy database by running the following.

```
python trainer.py --train-path data/databases/MP/formation_energy/train.csv --val-path data/databases/MP/formation_energy/val.csv --test-path data/databases/MP/formation_energy/test.csv --epochs 500 --batch-size 128 --train --test
```

### Structure-based Finder

An additional `--use-crystal-structure` flag is required to train structure-based `Finder` model. To train it for bandgap, you can run;

```
python trainer.py --train-path data/databases/MP/bandgap/train.csv --val-path data/databases/MP/bandgap/val.csv --test-path data/databases/MP/bandgap/test.csv --epochs 1200 --batch-size 128 --train --test --use-crystal-structure
```

### How to predict materials properties using a trained Finder model?

Once you train a `Finder` model, a directory named `saved_models/best_model_gnn` that contains the best model will be created. You may then run the following to make predictions using this pre-trained model. Use `--use-crystal-structure` flag if this is a structure-based `Finder` model.

```
python trainer.py --test-path data/databases/MP/formation_energy/test.csv --test
```

Prediction results will be saved in `results/` directory. 

## Questions and comments
Please contact a.a.ihalage@qmul.ac.uk or y.hao@qmul.ac.uk.

## Funding
We acknowledge funding received by The Institution of Engineering and Technology (IET) under the AF Harvey Research Prize. This work is supported in part by EPSRC Software Defined Materials for Dynamic Control of Electromagnetic Waves (ANIMATE) grant (No. EP/R035393/1) 
