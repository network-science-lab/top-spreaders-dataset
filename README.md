# ns-data-sources

Network Science Lab data sources. It requires to install `requirements.txt` containing version of used dvc package. Pulling the data for the first time requires authentication to Google Drive (`https://drive.google.com/drive/folders/0ACmD69K7LbU3Uk9PVA`) via an account that has permissions to our shared storage. For permissions, contact one of the contributors:

- `mateuszStolarski` 
- `anty-filidor` 
- `adampirog` 

## Pull the data
```bash
dvc pull <path_to_dvc_file>
```


## Add and push new data

```bash
dvc add <src>
dvc push <src>
```

