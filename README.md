# Installation Requirements

Apptainer, conda, and DVC must be installed on your system and in your path.

- [Apptainer installation guide](https://apptainer.org/docs/user/latest/quick_start.html#installation)
- [Conda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [DVC installation guide](https://dvc.org/doc/install)

This DVC pipeline will build the necessary conda environment using the provided `env.yaml` and `requirements.txt` files.

To reproduce the pipeline run the following command:
``` {bash}
dvc repro
```
