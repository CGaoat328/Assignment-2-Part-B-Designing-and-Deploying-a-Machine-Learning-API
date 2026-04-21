# Company Bankruptcy Risk Modeling API

## Project Overview

This project develops and deploys a machine learning API for the prediction task of **risk modeling**. The API predicts the **probability of company bankruptcy** using selected financial features from the **Company Bankruptcy Prediction** dataset. The model is served through **FastAPI**.

## Users

The main users of this service are:

- Financial analysts
- Banks and lenders
- Investors
- Business consultants and auditors
- Developers integrating the API into other systems

## Service Interaction Diagram

```text
User / Client Application
        ↓
      FastAPI
        ↓
 Logistic Regression Model
        ↓
   JSON Response
```


## Setup Instructions 

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run the API

```text
http://127.0.0.1:8000/docs
```

### 4. Open the API documentation

```bash
http://127.0.0.1:8000/docs
```


## API Documentation
This project uses FastAPI, which automatically generates interactive API documentation at:

/docs

The /docs endpoint shows the available endpoints, request body format, response format, and allows live testing of the API.



## Project Files

```text
README.md
train_model.py
app.py
requirements.txt
Dockerfile