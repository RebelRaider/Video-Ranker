# Video Ranking Model

This project aims to build a video ranking model using FAISS and CatBoostRanker. The model leverages Sentence Transformers for embedding video titles and uses a variety of features for ranking videos.

## Table of Contents

- [Video Ranking Model](#video-ranking-model)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Installing Dependencies with Poetry](#installing-dependencies-with-poetry)
    - [Activating the Virtual Environment](#activating-the-virtual-environment)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The project involves several key steps:

1. Downloading necessary data files from Yandex Disk.
2. Preparing the data by sampling and processing.
3. Setting up FAISS for efficient vector search.
4. Forming candidate sets and targets.
5. Engineering features from the available data.
6. Training a ranking model using CatBoostRanker.
7. Evaluating the model using various metrics.
8. Explaining model predictions using SHAP.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- Poetry for dependency management

### Installing Dependencies with Poetry

First, clone the repository:

```sh
git clone https://github.com/yourusername/video-ranking-model.git
cd video-ranking-model
```

Install `poetry` if you haven't already:

```sh
pip install poetry
```

Install the required Python packages using `poetry`:

```sh
poetry install
```

### Activating the Virtual Environment

Activate the virtual environment created by `poetry`:

```sh
poetry shell
```

Alternatively, you can run commands within the virtual environment using `poetry run`:

```sh
poetry run jupyter notebook video_ranking_model.ipynb
```

## Configuration

The configuration settings and constants are encapsulated in a `Config` class within the notebook. You can adjust the parameters as needed by modifying the class attributes.

```python
class Config:
    SEED = 42
    USE_FORMED_INDEX = True
    USE_FORMED_ID_MAPPING = True
    USE_FORMED_CANDIDATES = True
    VECTOR_DIM = 312
    BATCH_SIZE = 100_000
    SAMPLE_SIZE = 7_000_000
    TOP_N_QUERIES = 1000
    TOP_K = 300
    GENERATED_CANDIDATES_FILENAME = "generated_candidates.parquet"
    CANDIDATES_INDEX_FILENAME = "candidates.index"
    ID_MAPPING_FILENAME = "ind2videoid.json"
    MODEL_FILENAME = "ranker.ckpt"
    METRIC_PERIOD = 250
    QUANTILE_THRESHOLD = 0.999
    MODEL_PARAMS = {
        "task_type": "GPU",
        "verbose": False,
        "random_seed": SEED,
        "loss_function": "QueryRMSE",
        "learning_rate": 0.001,
        "l2_leaf_reg": 30,
        "iterations": 4000,
        "max_depth": 3,
        "metric_period": METRIC_PERIOD
    }
```

## Usage

1. Open the Jupyter Notebook:

jupyter notebook video_ranking_model.ipynb

1. Follow the steps in the notebook to:
   - Download data files.
   - Prepare and preprocess the data.
   - Set up FAISS index.
   - Form candidate sets and targets.
   - Engineer features.
   - Train the ranking model.
   - Evaluate the model.
   - Perform SHAP analysis.

## Project Structure

video-ranking-model/
│
├── video_ranking_model.ipynb     # Jupyter notebook with the entire workflow
├── pyproject.toml                # Poetry configuration file
├── poetry.lock                   # Poetry lock file with exact versions of dependencies
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
└── data/                         # Directory to store downloaded data files

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
