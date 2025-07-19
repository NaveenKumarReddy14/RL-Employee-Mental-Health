# Project Name
## RL-Based Employee Mental Health Monitoring System in Modern Workplace Environment



# Table of Contents
1.Overview

2.Features

3.Project Workflow

4.Tech Stack

5.Dataset

6.Usage

7.Results

7.Future Enhancements

## Quick Overview
Mental health issues are a growing concern, impacting individuals' well-being and productivity. This project introduces a Deep Quantum Network (DQN)-based Mental Health Prediction System, designed to assess users' mental health status based on their inputs through a Streamlit UI. Users provide responses related to factors such as work environment, personal history, and mental health-related experiences. The DQN model processes these inputs to generate a prediction, indicating whether the user may be experiencing mental health challenges. To enhance user engagement, the system provides dynamic visualizations based on the prediction results, offering insights into patterns and contributing factors affecting mental well-being. Additionally, a mental wellness chatbot is integrated, allowing users to seek guidance, self-care tips, and mental health resources. The chatbot serves as a supportive tool, offering proactive mental health assistance and recommendations. This project leverages cutting-edge AI, quantum computing principles, and deep learning techniques to improve mental health awareness and provide users with an interactive and informative experience. By combining predictive modeling, real-time visualization, and conversational AI, the system empowers individuals to take proactive steps toward mental well-being while fostering awareness and accessibility to mental health support
Keywords: Mental health prediction, Deep Q-Network (DQNN), Machine learning in healthcare, Real-time mental health assessment, Streamlit-based interface, AI-driven chatbot, Stress management.

## Features
Interactive Streamlit UI for data input and visualization.

Real-time prediction of mental health conditions.

Data cleaning, encoding, and preprocessing for accurate results.

Trained machine learning model for reliable predictions.

Lightweight and easy-to-use web app.

## Project Workflow

 # Mental Health Prediction System
![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/Flowchat.png?raw=true)

This project is a machine learning-based mental health prediction system that analyzes survey data to predict a user’s mental health condition. It uses a Streamlit web interface for user interaction and provides actionable suggestions based on the prediction.

Below is the step-by-step flow of the system:
![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/User%20input.png?raw=true)

User Input:

User enters survey data via the Streamlit UI.

Data Preprocessing:

The raw input data is cleaned, encoded, and transformed into numerical feature vectors.

Prediction Generation:

The pre-trained ML model (mental_health_model.pkl) processes the data and generates predictions.

## Display Results:

Predictions and relevant suggestions are displayed back on the Streamlit interface.

![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/unlikely%20prediction.png?raw=true)
![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/Likely%20prediction.png?raw=true)

Dataset Utilization:

A mental health dataset is used to train the ML model with features relevant to mental health analysis.


## Tech Stack
Programming Language: Python

Frontend/UI: Streamlit

Machine Learning: scikit-learn

Data Handling: Pandas, NumPy

Visualization: Plotly/Matplotlib (optional)

Model Storage: Joblib (mental_health_model.pkl)

## Dataset
The dataset includes survey responses related to mental health factors such as stress, anxiety, work-life balance, and other personal attributes.


## Usage
Launch the Streamlit app using streamlit run app.py.

Input your survey data into the provided fields.

Click the Predict button to get your mental health status and suggestions.

## Results
The model provides real-time mental health predictions.

Outputs are displayed along with recommendations for improving mental well-being.
![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/HeartRate%20%20with%20Recomendations.png)
![image alt](https://github.com/NaveenKumarReddy14/RL-Employee-Mental-Health/blob/main/Recomedations.png?raw=true)

## Future Enhancements
Integration of Reinforcement Learning for continuous improvement (dynamic suggestions).

Hybrid Quantum-Classical ML models for enhanced performance.

Adding a chatbot module for mental health assistance.
