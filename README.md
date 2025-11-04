# MLflow Credit Scoring Model Automation

[![CI/CD MLflow](https://github.com/lukaskrisna/mlflow-automation/actions/workflows/main.yml/badge.svg)](https://github.com/lukaskrisna/mlflow-automation/actions/workflows/main.yml)

A machine learning project that automates the training, tracking, and deployment of a credit scoring model using MLflow and GitHub Actions.

## Project Overview

This project implements an automated machine learning pipeline for credit scoring using:

- **MLflow** for experiment tracking and model management
- **Random Forest Classifier** for credit score prediction
- **GitHub Actions** for CI/CD automation
- **Docker** for model containerization and deployment

## Project Structure

```
mlflow-automation/
├── MLproject/
│   ├── conda.yaml          # Conda environment configuration
│   ├── MLproject            # MLflow project configuration
│   ├── modelling.py         # Main training script
│   ├── train_pca.csv        # Training dataset (PCA features)
│   └── train_pca_testing.csv # Testing dataset
├── .github/workflows/
│   └── main.yml            # CI/CD pipeline configuration
└── README.md               # This file
```

## Features

- **Automated Model Training**: Trains Random Forest classifier with configurable parameters
- **Experiment Tracking**: Uses MLflow to track experiments, metrics, and model artifacts
- **CI/CD Pipeline**: Automated training and deployment on every push/PR
- **Docker Integration**: Builds and deploys model as Docker container
- **Docker Hub Deployment**: Automatically pushes trained models to Docker Hub

## Technology Stack

- **Python 3.12.7**
- **MLflow 2.19.0** - Experiment tracking and model management
- **Scikit-learn 1.6.0** - Machine learning library
- **Pandas 2.2.3** - Data manipulation
- **NumPy 2.2.1** - Numerical computing
- **GitHub Actions** - CI/CD automation
- **Docker** - Containerization

## Model Details

- **Algorithm**: Random Forest Classifier
- **Target Variable**: Credit_Score (categorical: 0, 1, 2)
- **Features**: PCA-transformed features plus demographic data
- **Default Parameters**:
  - `n_estimators`: 505
  - `max_depth`: 35
- **Evaluation Metric**: Accuracy

## Setup and Installation

### Prerequisites

- Python 3.12.7
- MLflow
- Docker (for deployment)

### Local Development

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd mlflow-automation
   ```

2. **Create conda environment**:

   ```bash
   conda env create -f MLproject/conda.yaml
   conda activate mlflow-env
   ```

3. **Run the MLflow project**:

   ```bash
   mlflow run MLproject --env-manager=conda
   ```

4. **View MLflow UI**:
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000` to view experiments.

### Custom Parameters

You can customize model parameters:

```bash
mlflow run MLproject -P n_estimators=100 -P max_depth=20 -P dataset=train_pca.csv
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Environment Setup**: Installs Python 3.12.7 and dependencies
2. **Model Training**: Runs MLflow project with default parameters
3. **Model Packaging**: Builds Docker image with trained model
4. **Deployment**: Pushes Docker image to Docker Hub

### Required Secrets

Configure these secrets in your GitHub repository:

- `DOCKER_HUB_USERNAME`: Your Docker Hub username
- `DOCKER_HUB_ACCESS_TOKEN`: Your Docker Hub access token

## Docker Deployment

The trained model is automatically containerized and can be deployed using:

```bash
docker pull lukaskrisna/credit-scoring-model:latest
docker run -p 5000:8080 lukaskrisna/credit-scoring-model:latest
```

The model will be available at `http://localhost:5000/invocations` for predictions.

## Workflow Triggers

The CI/CD pipeline is triggered on:

- Push to `main` branch
- Pull requests to `main` branch

## Environment Variables

- `CSV_URL`: Path to the training dataset (`MLproject/train_pca.csv`)
- `TARGET_VAR`: Target variable name (`Credit_Score`)

## Troubleshooting

### Common Issues

1. **MLflow Run Fails**: Ensure all dependencies are installed correctly
2. **Docker Build Fails**: Check if the latest run_id exists in mlruns directory
3. **Docker Push Fails**: Verify Docker Hub credentials are correctly set in GitHub secrets
