# Welcome to the Onyx GenAI SDK

The goal of this project is to simplify the developer experience when interacting with Onyx GenAI Services. This project provides wrappers around the the underlying APIs provided by the service.

## Table of Contents

1. [Using the SDK in Onyx](#using-the-sdk-in-onyx)
2. [Embedding Client](#embedding-client)
3. [Model Client](#model-client)

## Using the SDK in Onyx

1. Create a Conda Store Environment with all dependencies listed in the requirements.txt

2. Start your JupyterLab Server

3. Create a new Jupyter Notebook

4. Add the onyxgenai imports to your project. See the notebooks section of this project for example usage

## Embedding Client

The Embedding Client provides access to the Onyx GenAI Embedding Service. The client provides access to functionality such as:

- Generating Text and Image Embeddings and Vector Storage
- Retrieving Vector Store Collections
- Vector Database Search

## Model Client

The Model Client provides access to the Onyx GenAI Model Store Service. The client provides access to functionality such as:

- Retrieving Model Info
- Retrieving Active Model Deployment Info
- Deploying and Deleting Model Deployments
- Performing Text and Image Prediction and Embedding
- Generating Text Completions from an LLM
