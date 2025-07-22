# TopSpreadersDataset

This repository contains a dataset of multilayer networks and the spreading potentials of their  
actors. It also includes a Python package to facilitate the data loading process. The dataset is one  
of the artefacts described in the paper  
[*Identifying Super Spreaders in Multilayer Networks*](https://arxiv.org/abs/2505.20980).

- **Authors**: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski  
- **Affiliation**: Wrocław University of Science and Technology, Wrocław, Lower Silesia, Poland

## Structure of the Repository

```bash
.
├── .dvc                      -> DVC configuration files
├── env                       -> Env. requirements for the Python package
├── nsl_data_utils            -> Python package for handling the dataset
├── nsl_data_sources          -> Directory with source data files
└── README.md
```

## Source Data Files

The dataset is managed using [DVC](https://dvc.org/). To use it, first install the required  
dependencies listed in `requirements.txt`, including the appropriate version of DVC.

### Full Access

To download the dataset, you must authenticate with a Google account that has access to the shared
Google Drive storage: `https://drive.google.com/drive/folders/0ACmD69K7LbU3Uk9PVA`. If you need
access, please contact one of the contributors. Then, to fetch the data, run `dvc pull`.

### Paper Version

A public DVC configuration for the dataset version used in the paper is available at:
`https://drive.google.com/drive/folders/0ACmD69K7LbU3Uk9PVA`. To use it, unpack the archive, and
move the folder `cache` into the `.dvc` directory. Then, checkout to the tag `2.0.1` and
execute: `dvc pull`. 

## Using the Package

To install the package in editable mode, run:

```bash
pip install -e nsl_data_utils
```

To contribute to the repository, create and activate the development environment using Conda:

```bash
conda env create -f env/conda.yaml
conda activate nsl-data-utils
```
