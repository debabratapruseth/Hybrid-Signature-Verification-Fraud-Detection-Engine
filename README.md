# Signature Forensic Analysis

A hybrid offline handwritten signature verification system that combines **Deep Learning**, **Structural AI**, and **Document Forensics** to support explainable signature verification and review prioritization.

# Architecture and Pipeline 

![](https://github.com/debabratapruseth/Signature-Forensic-Analysis/blob/main/Architecture.png)

# Sample Output Analysis 

![](https://github.com/debabratapruseth/Signature-Forensic-Analysis/blob/main/Sample%20Output.png)

# Features

- PDF and document preprocessing 

- YOLO-based signature detection

- SAM-assisted signature segmentation ( For dirty signatures) 

- OpenCV signature cleaning

- Deep learning signature verification (ResNet-18 Siamese Network)

- Structural signature comparison

- Document forensics screening

- Reliability-aware evidence fusion

- LLM Explainable technical and stakeholder reports

# Three Layer Forensic Analysis 

### Branch 1 — Learned Verification

- ResNet-18 Siamese Network

- Signature embedding generation

- Cosine similarity

- External validation

- Explainable verification evidence

### Branch 2 — Structural AI

- Skeleton extraction

- Shape Context

- Fourier descriptors

- Hu Moments

- Contour analysis

- Graph-based comparison

- Structural similarity

### Branch 3 — Document Forensics

- Copy-paste screening

- ORB feature matching

- Template matching

- Error Level Analysis (ELA)

- Compression analysis

- Noise consistency

- Reference reuse detection


### Output


- Executive stakeholder report

- Technical fusion report

- Evidence dashboard

- Reliability-aware fusion summary



# How to Use

This project is designed to run on **Google Colab** using a **GPU runtime**, with **Google Drive** used for storing datasets, model checkpoints, and outputs.
> **Note:** You can refactor the project to run locally using **VS Code** and **GitHub**. However, a GPU is recommended for training the deep learning models.

## Prerequisites
- Google Account
- Google Drive
- Google Colab
- OpenAI API Key (for report generation)

## Step 1: Prepare the Project
1. Extract **`Signature-Forensic-Prototype.zip`** into your Google Drive.
2. Download dataset from internet and store in Google Drive for training
   - https://docs.ultralytics.com/datasets/detect/signature for YOLO training
   - HSig260 dataset for Siamese ResNet-18 training

## Step 2: Train the Models (One-Time Setup)
Open:
```text
colab_working/train_models_cells.ipynb
```
Run all cells to train the required models:
- YOLO Signature Detector
- ResNet-18 Siamese Signature Verification Model
The trained model checkpoints will automatically be saved for future use.
> This step only needs to be performed once unless you wish to retrain the models.

## Step 3: Run a Signature Verification Case
Open:
```text
colab_working/complete_signature_case_runner.ipynb
```
The notebook will prompt you to upload:
- 1 Questioned Signature
- 3 Reference Signatures

The pipeline will automatically execute all three verification branches:
- Branch 1 – Learned Signature Verification
- Branch 2 – Structural Signature Comparison
- Branch 3 – Document Forensics

The final stage uses an **OpenAI LLM** to generate:
- Executive Stakeholder Report
- Technical Fusion Report
Before running this step, provide your OpenAI API key or configure it as an environment variable.


## License

MIT License
