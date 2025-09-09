# TopSpreadersDataset

This repository contains a dataset of multilayer networks and the spreading potentials of their  
actors. It also includes a Python package to facilitate the data loading process. The dataset is one  
of the artefacts described in the paper
[*Identifying Super Spreaders in Multilayer Networks*](https://arxiv.org/abs/2505.20980).

- **Authors**: Michał Czuba, Mateusz Stolarski, Adam Piróg, Piotr Bielak, Piotr Bródka
- **Affiliation**: Wrocław University of Science and Technology, Wrocław, Lower Silesia, Poland

## Functionality

The dataset comprises over 200 multilayer networks, including both synthetic and real-world
examples. Each actor is labelled according to their spreading capability, assessed through
simulation. Specifically, for every actor, a diffusion under the Multilayer Independent Cascade
Model is initiated with that actor as the sole seed. From each simulation, a feature vector is
extracted, containing:
- the total number of activated nodes,
- the duration of the diffusion process (i.e. number of time steps),
- the maximum number of activations in a single step (the peak),
- the time step at which this peak occurs.

A summary of the networks included in the dataset, along with their key statistics, is provided
below. For synthetic networks, mean values are reported across all instances.

| Network type     | Layers | Actors  | Nodes   | Edges    | Degree |
| ---------------- | ------ | ------- | ------- | -------- | ------ |
| artificial-er    | 3.52   | 558.19  | 1741.70 | 6684.00  | 24.13  |
| artificial-pa    | 3.52   | 574.51  | 1976.07 | 42636.53 | 122.10 |
| artificial-small | 2.75   | 1000.00 | 2750.00 | 6609.12  | 13.22  |
| arxiv            | 13     | 14065   | 26796   | 59026    | 8.39   |
| aucs             | 5      | 61      | 224     | 620      | 20.33  |
| ckmp             | 3      | 241     | 674     | 1370     | 11.37  |
| eu-trans         | 37     | 417     | 2034    | 3588     | 17.21  |
| l2-course        | 2      | 41      | 82      | 297      | 14.49  |
| lazega           | 3      | 71      | 212     | 1659     | 46.73  |
| timik            | 3      | 61702   | 102247  | 875191   | 28.37  |

## Structure of the Repository

```bash
.
├── .dvc                      -> DVC configuration files
├── env                       -> Env. requirements for the Python package
├── tsds_utils                -> Python package for handling the dataset
├── tsds_sources              -> Directory with source data files
└── README.md
```

Version 2.0.0 - a dataset in the exact form that was used in experiments
Version 2.0.1 - a variant with improved manual

## Source Data Files

The dataset is managed using [DVC](https://dvc.org/). To use it, first install the required  
dependencies listed in `requirements.txt`, including the appropriate version of DVC.

### Full Access

To download the dataset, you must authenticate with a Google account that has access to the shared
Google Drive storage: `https://drive.google.com/drive/folders/0ACmD69K7LbU3Uk9PVA`. If you need
access, please contact one of the contributors. Then, to fetch the data, run `dvc pull`.

### Paper Version

A public DVC configuration for the dataset version used in the paper is available at:
`https://drive.google.com/file/d/1OnNLhjKotOlV0c2_1FcJgCCzQdTGRtoQ`. To use it, unpack the archive,
and move its contents into the `.dvc` directory of this project. Then, execute: `dvc checkout`. 

## Using the Package

To install the package in editable mode, run:

```bash
pip install -e .
```

To contribute to the repository, create and activate the development environment using Conda:

```bash
conda env create -f env/conda.yaml
conda activate tsds-utils
```

## Acknowledgment

This work was supported by the National Science Centre, Poland [grant no. 2022/45/B/ST6/04145]
(www.multispread.pwr.edu.pl); the Polish Ministry of Science and Higher Education programme
“International Projects Co-Funded”; and the EU under the Horizon Europe [grant no. 101086321].
Views and opinions expressed are those of the authors and do not necessarily reflect those of
the funding agencies.
