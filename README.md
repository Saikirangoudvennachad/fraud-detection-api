# Real-Time Fraud Detection API 

A production-ready Machine Learning inference service designed to detect fraudulent transactions in real-time. This project demonstrates an end-to-end MLOps workflow: from training a model on imbalanced data to deploying it as a containerized microservice.

##  Key Features
* High-Performance Inference: Built with FastAPI for low-latency response times (<50ms).
* Imbalanced Data Handling: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in fraud datasets.
* Robust Model: Trained using RandomForest/XGBoost for high precision.
* Production-Ready: Containerized with Docker for consistent deployment across environments.
* Type Safety: Uses Pydantic for strict input data validation.

## Tech Stack
* Language: Python 3.9
* Framework: FastAPI, Uvicorn
* ML Libraries: Scikit-learn, Pandas, Imbalanced-learn (SMOTE), Joblib
* Containerization: Docker
* Tools: Git, Swagger UI

##  Project Structure
```bash
├── app.py               # FastAPI application (Inference Service)
├── train_model.py       # ML Pipeline: Data Gen -> SMOTE -> Training -> Saving
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── fraud_model.joblib   # Serialized model artifact (generated after training)
└── README.md            # Project documentation
