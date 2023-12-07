
## MLFlow
- Documentation: https://mlflow.org/docs/latest/index.html

### Set Backend and Artefacts stores 

This supposes that we are using `mlflow server` command.

Type `mlflow server --help` command to see options.

#### Backend store
As per the official documentation: The location (`file`or `database`) where MLFlow stores experiments and run metadata as well as metrics, parameters and tags for runs.

- Configure the `backend store`: use `--backend-store-uri` option
  - file store: `--backend-store-uri </PATH/TO/FILE/STORE>` or `--backend-store-uri file:/<PATH/TO/FILE/STORE>`
    - Set by default to `./mlruns` directory 
  - database store: `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`
    - Supported databases: `MysQL, PostgreSQL, SQLite, ...`,  
    - Example with local SQLite database: `--backend-store-uri sqlite:///mlflow.db`

#### Artefacts store
The location where MLFlow stores artefacts (models, datasets, ...)

- Configure `artefact store`: use `--default-artifact-root` option. The artefact store supports local file path or remote storage system like Amazon S3, Azure Blob Storage, Google Cloud Storage, SFTP server, ...
  - Set by default to `./mlruns` directory
  - Example with Amazon S3: `--default-artifact-root s3://my-root-bucket`
  - Example with Azure Blob Storage: `--default-artifact-root wasbs://<container>@<storage-account>.blob.core.windows.net/<path>`


### Tracking UI
The [tracking UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui) is a web-based UI that helps to visualize, 
search, compare runs, and download run artifacts (files, models, datasets, ...), ...

How to get access to the UI ?
- Local artefacts storage (e.g. runs logged into a local `mlruns` directory): run `mlflow ui` command in the directory above it and go to the `http://localhost:5000/` url
- Remote artefacts storage: Go to `http://<ip address of your MLflow tracking server>:5000` url 

### Recap


|                   | Tracking URI                                          | Registry URI                                          | Difference                                           |
|-------------------|-------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Purpose**       | Stores experiment and run metadata (e.g., parameters, metrics, tags)                    | Stores registered models and model versions           | Registry URI is required for model registry features  |
| **Usage**         | Can be HTTP/HTTPS URI, database connection string, or local path | Can be the same as tracking URI or a different location supporting the same backend store types    | If not using the model registry, only tracking URI needs to be set |
| **Setting**       | `mlflow.set_tracking_uri` or `MLFLOW_TRACKING_URI`    | `mlflow.set_registry_uri` or `MLFLOW_REGISTRY_URI`    |                                                       |


    - The main difference between tracking URI and registry URI is that the registry URI is only required if we want to use the model registry features, such as registering models and transitioning stages. The model registry allows the management of the full lifecycle of MLflow models. If we do not use the model registry, we would only need to set the tracking URI.


## SQLite
- Tutorial: https://www.sqlitetutorial.net/


## How to

### Set mlflow server
Let's test two simple scenarios. See [documentation](https://mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded) for more use cases.


#### Option 1: local file store as backend store and local file store as artefacts store. 
- Use default settings (i.e., artefact store and the backend store are not specified): the `backend` and `artifact` store share a directory on the local filesystem: `./mlruns` by default 
  - Run `mlflow ui` command from the directory above `mlruns` directory
- Specify artefact store and the backend store
  - Let `artefacts` directory to be the artefact store and `runs_dir` directory to be the backend store.
    - Run `mlflow serevr --backend-store-uri <PATH/TO/runs_dir> --default-artifact-root <file:/PATH/TO/artefacts>`

  
#### Option 2: Local SQlite as backend store and local file store as artefact store
    
- [Documentation](https://mlflow.org/docs/latest/tracking.html#scenario-2-mlflow-on-localhost-with-sqlite)


```shell
mlflow server --backend-store-uri sqlite:///<PATH/TO/MYDataBase.db> --default-artifact-root <file:/PATH/TO/ARTEFACT/STORE>
```

#### Example with option 2
- Let `mlflow.db` to be the local SQLite DB backend store and `artefacts` to be the local file store for artefact store and where `mlflow.db` and `artefacts` are in `mlflow` directory.

1. Run the following command from `mlflow` directory

```shell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:artifacts --host 0.0.0.0 --port 5000
````
Or (with the use of `nohup` command when we need to run `mlserver` in background)

```shell
nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:artefacts --host 0.0.0.0 --port 5000 > nohup.out &
```

2. Tracking URI setting
    - Set `MLFLOW_TRACKING_URI` environment variable to `http://localhost:5000` or `mlflow.set_tracking_uri("http://localhost:5000")` when using mlflow API.
        - Remark: `mlflow server` command need to be run with the correct parameters before, otherwise the following error may be risen.
```text
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=5000): Max retries exceeded with url: /api/2.0/mlflow/experiments/get-by-name?experiment_name=mlops-exp (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f669f365ed0>: Failed to establish a new connection: [Errno 111] Connection refused'))
```

3. Run python scripts  (by with different parameters' values)

    - How to set arguments: `python train_1.py --help`

```python
python train_1.py --alpha 1 l1-ratio 0 --intercept true --max-iter 500
```
```python
python train_1.py --alpha 1 l1-ratio 1 --intercept true --max-iter 100
```
```python
python train_1.py --alpha 1 l1-ratio 0.5 --intercept true --max-iter 100
```
