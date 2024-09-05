# Network Science Lab Data Sources 

A repository with dataset and Python package to handle it.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Structure of the repository

```bash
.
├── .dvc               -> configuration of DVC
├── env                -> a definition of the runtime environment for Python package
├── infmax_data_utils  -> Python package handling data with spreading models used to generate it
├── ns-data-sources    -> directory with datasets
├── mln_potentials_eda.ipynb  -> jupyter notebook to create EDA of datasets
└── README.md          -> main entrypoint to trigger the pipeline
```

## The dataset

Dataset is stored with DVC. It requires to install `requirements.txt` containing version of used dvc
package. Pulling the data for the first time requires authentication to Google Drive
(`https://drive.google.com/drive/folders/0ACmD69K7LbU3Uk9PVA`) via an account that has permissions
to our shared storage. For permissions, contact one of the contributors. Then, to download the data,
execute in a shell: `dvc pull <path_to_dvc_file>`.

## Python package

`infmax_data_utils`is Python package with loaders for the datasets and implementations of 
Independent Cascade Model used to generate spreading potentials the agents. To use it, you can
install it with command:

```bash
pip install -e infmax_data_utils
```

To contribute, please install first Conda environment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-data-utils
```
