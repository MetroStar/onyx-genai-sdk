# Onyx GenAI SDK

The goal of this project is to simplify the developer experience when interacting with Onyx GenAI Services. This project provides wrappers around the the underlying APIs provided by the service.

## Table of Contents

1. [Installation](#installation)
2. [Using the SDK in Onyx](#using-the-sdk-in-onyx)
3. [Running the Project Locally](#running-the-project-locally)
4. [Running Unit Tests](#running-unit-tests)
5. [Running Code Quality Checks](#running-code-quality-checks)

## Installation

To install with `pip`, run:

```bash
pip install onyxgenai
```

To install with `conda`, run:

```bash
conda install -c conda-forge onyxgenai
```

## Using the SDK in Onyx

1. Create a Conda Store Environment with all dependencies listed in the requirements.txt

2. Start your JupyterLab Server

3. Create a new Jupyter Notebook

4. Install the SDK as indicated above

5. Add the onyxgenai client imports to your project

For more in depth examples, see [notebooks](https://github.com/MetroStar/onyx-genai-sdk/tree/main/notebooks) section of this repo.

### Embedding Client

The Embedding Client provides access to the Onyx GenAI Embedding Service. The client provides access to functionality such as:

- Generating Text and Image Embeddings and Vector Storage
- Retrieving Vector Store Collections
- Vector Database Search

### Model Client

The Model Client provides access to the Onyx GenAI Model Store Service. The client provides access to functionality such as:

- Retrieving Model Info
- Retrieving Active Model Deployment Info
- Deploying and Deleting Model Deployments
- Performing Text and Image Prediction and Embedding
- Generating Text Completions from an LLM

## Running the Project Locally

1. To create an environment, run the following:

```sh
virtualenv -p python3 venv
source venv/bin/activate
```

2. To install dependencies, run the following:

```sh
pip install .
pip install -e ".[dev]" # Optional
```

## Running Unit Tests

1. To run unit tests, run the following:

```sh
pytest
```

## Running Code Quality Checks

1. To run code quality checks, run the following:

```sh
ruff check .
```
