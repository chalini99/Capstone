# 🌱 AGROVISION AI

## AI-Powered Crop Disease Detection and Smart Agrochemical Recommendation System

**An intelligent agricultural decision-support system for early crop disease identification and informed disease management.**

---

## 1. About the Project

Agriculture plays a vital role in food security and economic development. However, crop diseases continue to affect agricultural productivity, resulting in significant losses for farmers. Identifying diseases at an early stage is essential, but traditional disease identification often depends on manual observation and agricultural expertise.

**AGROVISION AI** aims to address this challenge through artificial intelligence and deep learning.

The project introduces an intelligent system that analyzes images of crop leaves, identifies potential diseases using a trained deep learning model, and provides relevant disease information and management recommendations.

By combining image-based disease classification with a structured agricultural knowledge base, AGROVISION AI aims to make crop disease information more accessible and support informed agricultural decision-making.

## 2. Problem Statement

Farmers often face difficulties identifying crop diseases accurately and selecting appropriate disease management practices. Delayed identification, limited access to expert guidance, and a lack of reliable disease-specific information can contribute to crop damage and reduced agricultural productivity.

There is a need for an intelligent, accessible system that can identify potential crop diseases from leaf images and provide relevant management and preventive guidance.

## 3. Proposed Solution

AGROVISION AI integrates deep learning-based image classification with a recommendation engine to provide an end-to-end crop disease analysis workflow.

The system:

1. Accepts a crop leaf image from the user.
2. Preprocesses the uploaded image for model prediction.
3. Uses an EfficientNet-B0 deep learning model to classify the image into one of 38 classes.
4. Identifies the predicted crop and its potential disease condition.
5. Retrieves relevant disease information from a structured knowledge base.
6. Provides disease-specific management guidance and preventive measures.
7. Uses Grad-CAM to visualize important image regions influencing the model prediction.

## 4. Key Features

### 🔬 AI-Based Crop Disease Detection

Uses a trained EfficientNet-B0 model to analyze leaf images and classify crop health conditions across 38 classes.

### 🌿 Smart Recommendation Engine

Retrieves relevant disease information and provides chemical, biological, and cultural management guidance where available in the knowledge base.

### 🧠 Explainable AI with Grad-CAM

Generates visual heatmaps to help users understand which regions of a leaf image influenced the model's prediction.

### 📚 Disease Information

Provides information about identified diseases, including descriptions, symptoms, causes, and preventive measures.

### 📊 Crop Health Analytics

Presents visual summaries of recorded crop disease analyses through an interactive analytics dashboard.

### 🗂️ Analysis History

Maintains previous analysis records so users can review their past crop disease assessments.

### 📄 PDF Report Generation

Generates downloadable reports containing crop disease analysis information.

## 5. System Workflow

```text
        USER
         |
         v
   Upload Leaf Image
         |
         v
  Image Preprocessing
         |
         v
  EfficientNet-B0 Model
         |
         v
  Disease Classification
         |
         v
  Crop & Disease Identification
         |
         +----------------------+
         |                      |
         v                      v
   Grad-CAM             Knowledge Base
   Visualization        Disease Information
                               |
                               v
                      Recommendation Engine
                               |
                               v
                  Management & Prevention
                         Guidance
                               |
                               v
                    Results & PDF Report
```

## 6. Technology Stack

| Layer                         | Technologies    |
| ----------------------------- | --------------- |
| Programming Language          | Python          |
| Deep Learning Framework       | PyTorch         |
| Model Architecture            | EfficientNet-B0 |
| Image Processing              | OpenCV, Pillow  |
| Explainable AI                | Grad-CAM        |
| Frontend / User Interface     | Streamlit       |
| Data Visualization            | Plotly          |
| Recommendation Knowledge Base | JSON            |
| Database                      | SQLite          |
| Testing                       | Pytest          |

## 7. Deep Learning Model

The project uses **EfficientNet-B0**, a convolutional neural network architecture designed to achieve an effective balance between model performance and computational efficiency.

The trained model is configured to classify images into 38 crop disease and healthy-condition classes.

**Model details:**

- Model: EfficientNet-B0
- Framework: PyTorch
- Classification categories: 38
- Model checkpoint: `models/efficientnet_b0_best.pth`
- Output: Predicted class and confidence score

## 8. Recommendation Engine

The recommendation engine connects the model's disease prediction with a structured JSON-based agricultural knowledge base.

It retrieves available information related to:

- Disease description
- Symptoms
- Causes
- Chemical management
- Biological management
- Cultural management
- Preventive measures
- References and supporting information

The system displays available information rather than treating missing or unverified treatment information as confirmed recommendations.

## 9. Explainable AI

AGROVISION AI incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of deep learning predictions.

Grad-CAM generates a heatmap highlighting image regions that contributed to the model's classification decision.

This helps users interpret model predictions rather than relying exclusively on a classification label.

## 10. Project Structure

```text
Capstone_Project/
│
├── app.py
│
├── assets/
│
├── config/
│   └── class_mapping.json
│
├── database/
│
├── knowledge_base/
│   └── disease_treatments.json
│
├── models/
│   └── efficientnet_b0_best.pth
│
├── modules/
│   ├── disease_detection.py
│   ├── recommendation_engine.py
│   ├── gradcam.py
│   ├── image_preprocessing.py
│   ├── disease_information.py
│   ├── validators.py
│   └── ui_*.py
│
├── sample_images/
│
├── tests/
│
├── utils/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 11. Installation and Execution

### Prerequisites

- Python 3.11
- Git
- pip

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Capstone_Project
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

For Windows:

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will be available in your local browser.

## 12. Project Objectives

- Develop an AI-based system for identifying crop diseases from leaf images.
- Implement a deep learning model for multi-class crop disease classification.
- Build a knowledge-based recommendation engine for disease management.
- Integrate explainable AI to improve the interpretability of model predictions.
- Provide an interactive dashboard for crop disease analysis and historical records.
- Support informed agricultural decision-making through accessible disease information.

## 13. Future Scope

- Expand the disease knowledge base with more verified agricultural information.
- Improve model performance using diverse real-world crop images.
- Support additional crops and disease categories.
- Integrate multilingual support for improved accessibility.
- Explore mobile-based deployment for field-level usage.
- Collaborate with agricultural experts to validate recommendations.

## 14. Important Disclaimer

AGROVISION AI is an agricultural decision-support prototype. Model predictions are preliminary screening results and should not be treated as confirmed diagnoses. Users should consult qualified agricultural experts before applying crop treatments. Chemical management information must be verified against locally approved agricultural guidance.

---

---

*AGROVISION AI — Bringing Artificial Intelligence into Crop Health Monitoring.*
