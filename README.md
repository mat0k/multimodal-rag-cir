# multimodal-rag-cir

This repository contains the implementation for my Master’s thesis project on **Enhancing Multimodal RAG Systems through Cross-Modal Retrieval and Reranking**, with a focus on **Composed Image Retrieval (CIR)** tasks.

The project investigates how a **retriever + reranker pipeline** can improve multimodal retrieval performance compared to encoder-only retrieval approaches.

## Project Goal

The main objective is to build and evaluate a pipeline consisting of:

1. **Retriever** a dual-encoder model used to retrieve candidate images from a dataset.
2. **Reranker** a multimodal cross-encoder used to refine the top retrieved results.
3. **Distillation** transferring knowledge from the two-stage system into a lightweight retriever.

The retrieval task follows the **Composed Image Retrieval (CIR)** setting, where a query is composed of:

* a **reference image**
* a **text modification**

The system must retrieve the image that matches the modified concept.

Example:

Reference image: red dress
Text: *"make it sleeveless"*

The model should retrieve images of **sleeveless red dresses**.

## Datasets

The main datasets used in this project are:

* **CIRR**
* **FashionIQ**

These datasets are designed specifically for composed image retrieval tasks.

## Current Phase

The current stage of the project focuses on:

* selecting a suitable **retriever backbone**
* setting up the **project repository**
* integrating dataset loading and **retrieval evaluation scripts**
* running **baseline retrieval experiments**

## Repository Structure (initial)

```
multimodal-rag-cir/
│
├── src/                # core implementation
│   ├── datasets/       # dataset loaders (CIRR, FashionIQ)
│   ├── retrievers/     # retriever models
│   ├── evaluation/     # retrieval evaluation metrics
│   └── utils/          # helper utilities
│
├── scripts/            # experiment scripts
│
├── configs/            # configuration files
│
└── data/               # local data symlink (ignored in git)
```

## Notes

This repository is designed to remain **modular and extensible**, allowing future integration of:

* multimodal rerankers
* knowledge distillation methods
* additional retrieval models
* extended evaluation experiments.
