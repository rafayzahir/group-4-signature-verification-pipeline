# Signature Verification System

An AI-powered Signature Verification System that compares a reference signature with a signature found in a document using Deep Learning and Computer Vision.

## Overview

The Signature Verification System is designed to determine whether a signature in a document matches a given reference signature.

The system uses signature detection and deep learning-based similarity comparison to generate a similarity score and verification result.

It provides a web-based interface where users can upload signature images and receive verification results.

## Key Features

- **Signature Upload:** Upload a reference signature and a document for verification.
- **Signature Detection:** Detect signature regions in the uploaded document.
- **Deep Learning-Based Comparison:** Extract signature embeddings using a ResNet50-based model.
- **Similarity Score:** Calculate similarity using Cosine Similarity and display a score from 0 to 100.
- **Verification Result:** Compare the similarity score against a threshold of 60.
- **Background Processing:** Use Redis and Celery to process verification tasks asynchronously.
- **REST API:** Use FastAPI to handle verification requests and retrieve job results.
- **Experiment Tracking:** Track model experiments and evaluation metrics using Comet ML.

## System Architecture

The system follows a frontend-backend architecture with asynchronous background processing.

```text
                  User
                   |
                   v
             React Frontend
                   |
                   v
             FastAPI Backend
                   |
                   v
             Redis Message Broker
                   |
                   v
             Celery Worker
                   |
          ---------------------
          |                   |
          v                   v
   Signature Detection   Similarity Model
       (ONNX)               (ResNet50)
          |                   |
          ---------------------
                   |
                   v
          Similarity Score
                   |
                   v
          Verification Result
                   |
                   v
             React Frontend
```

## Technologies Used

| Component | Technology |
|---|---|
| Frontend | React.js |
| Backend | FastAPI |
| Programming Language | Python |
| Deep Learning | PyTorch |
| Feature Extraction | ResNet50 |
| Similarity Measurement | Cosine Similarity |
| Signature Detection | ONNX Runtime |
| Background Tasks | Celery |
| Message Broker | Redis |
| Experiment Tracking | Comet ML |
| Version Control | Git & GitHub |

## How It Works

1. The user uploads a reference signature and a document.
2. The backend receives the verification request.
3. A background task is submitted through Redis to Celery.
4. The detection component identifies the signature region in the document.
5. The ResNet50-based similarity model extracts feature embeddings.
6. Cosine Similarity calculates the similarity between the two signatures.
7. The score is converted into a percentage between 0 and 100.
8. The system compares the score against the configured threshold of 60.
9. The verification result is returned to the frontend.

## Similarity Verification

The similarity model uses a Siamese-style approach with a shared ResNet50 feature extractor.

- Genuine signature pairs are trained to have similar embeddings.
- Forged signature pairs are trained to have different embeddings.
- Contrastive Loss is used during model training.
- Cosine Similarity is used to calculate the similarity score.

**Current verification threshold:** 60

A score of 60 or above is classified as a match under the current threshold configuration. A score below 60 is classified as a non-match.

> Note: The threshold is a configurable decision rule and should be validated against representative evaluation data before real-world use.

## Backend API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Check backend health |
| POST | `/verify` | Submit a signature verification request |
| GET | `/status/{job_id}` | Check verification task status |
| GET | `/result/{job_id}` | Retrieve verification result |

### API Documentation

When the backend is running, open:

`http://127.0.0.1:8000/docs`

This opens the interactive Swagger API documentation.

## Project Setup

### Prerequisites

Make sure the following tools are installed:

- Python 3.10 or later
- Node.js and npm
- Docker
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/rafayzahir/group-4-signature-verification-pipeline.git
```

```bash
cd group-4-signature-verification-pipeline
```

### 2. Frontend Setup

Navigate to the React frontend directory:

```bash
cd frontend-react
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

### 3. Backend Setup

Navigate to the backend directory containing `requirements.txt`.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server from the directory containing the `app` package:

```bash
uvicorn app.main:app --reload
```

Backend URL:

`http://127.0.0.1:8000`

### 4. Redis Setup

Start Redis using Docker:

```bash
docker start sig-redis
```

Check the Redis connection:

```bash
docker exec sig-redis redis-cli ping
```

Expected output:

```text
PONG
```

For a fresh setup, create a Redis container and configure the Redis connection URL in the application environment.

### 5. Celery Worker Setup

Start the Celery worker from the worker project directory, where `tasks.py` is located:

```bash
celery -A tasks worker --loglevel=info --pool=solo
```

Keep the worker running while processing verification tasks.

> Note: The worker and backend must use compatible Redis broker and result-backend configurations.

## Experiment Tracking

Comet ML is used to organize model experiments and monitor training and evaluation metrics.

The experiment tracking includes:

- Training loss
- Test AUC
- Genuine signature similarity
- Forged signature similarity
- Training hyperparameters
- Training duration

## Model Files

Large trained model files should be managed separately from the source code repository when appropriate.

The application requires its configured detection and similarity model files to be available to the backend or worker.

Make sure the correct model paths and environment variables are configured before running verification.

## Project Structure

The repository contains the frontend and backend components. The worker and model files may be maintained separately depending on the final team repository structure.

```text
group-4-signature-verification-pipeline/
│
├── frontend-react/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── similarity/
│   └── ...
│
├── README.md
├── .gitignore
└── LICENSE
```

## Current Development Status

- [x] Similarity approach selected
- [x] Similarity model training completed
- [x] Similarity score function implemented
- [x] Verification threshold configured
- [x] Frontend upload and result interface developed
- [x] FastAPI verification backend added
- [x] Redis and Celery configured for local processing
- [ ] Experiment tracking evidence and evaluation visualizations
- [ ] Complete end-to-end testing
- [ ] Production deployment

## Future Improvements

- Improve model performance with additional signature datasets.
- Evaluate the verification threshold using representative test data.
- Improve detection and similarity evaluation.
- Deploy the complete system to a cloud server.
- Add secure authentication and user management.
- Improve logging, monitoring, and error handling.

## Team Project

**Project:** Signature Verification System

**Repository:** [Group 4 – Signature Verification Pipeline](https://github.com/rafayzahir/group-4-signature-verification-pipeline)

This project demonstrates the integration of Computer Vision, Deep Learning, REST APIs, and asynchronous task processing to build an AI-based signature verification application.

---

*Developed as a collaborative AI/ML and Software Engineering project.*
