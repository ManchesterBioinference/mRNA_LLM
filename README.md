# mRNA Ribosome Density Prediction

This project predicts mRNA ribosome density (a.k.a. translation efficiency or TE) and decay rates in *Drosophila melanogaster* using a fine-tuned RNA Language Model (LLM). It integrates sequence information from 3' UTRs with additional features like codon usage and RNA secondary structure stability.

## Publication

- **Preprint**: A preprint describing this work is available on bioRxiv: [10.64898/2025.12.04.692303v1](https://www.biorxiv.org/content/10.64898/2025.12.04.692303v1)

## Project Overview

- **Goal**: Predict ribosome density and mRNA decay from sequence data.
- **Model**: Extended the pretraining of GenaLM Fly (BERT-based) on 5' & 3' UTR pairs and fine-tuned the model with a regression head.
- **Features**:
  - 5' & 3' UTR sequences
  - Codon usage metrics
  - Minimum Free Energy (MFE) from RNA folding (LinearFold)
  - GC content and sequence length
- **Pipeline**: Managed by DVC for reproducibility, covering data download, preprocessing, feature extraction, and model training.

- **Decay analysis**: Detailed decay-rate analyses are kept on the `decay` branch of this repository (see the `decay` branch for notebooks and results).

## Installation Requirements

Apptainer, conda, and DVC must be installed on your system and in your path.

- [Apptainer installation guide](https://apptainer.org/docs/user/latest/quick_start.html#installation)
- [Conda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [DVC installation guide](https://dvc.org/doc/install)

## Usage

This DVC pipeline will build the necessary conda environment using the provided `environment.yaml`.

To reproduce the pipeline run the following command:

``` {bash}
dvc repro
```

## Repository Structure

- `dvc.yaml`: Pipeline definition.
- `params.yaml`: Configuration parameters.
- `scripts/`: Source code for data processing and training.
- `notebooks/`: Exploratory analysis and visualization.
