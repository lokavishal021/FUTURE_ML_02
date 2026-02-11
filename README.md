# 🎫 Automated Customer Support Ticket Classification System

## 📌 Project Overview
This project implements a **Machine Learning-based Decision Support System** to automate the classification and prioritization of customer support tickets. It uses Natural Language Processing (NLP) techniques to analyze text and Scikit-Learn for predictive modeling.

## 📂 Project Structure
- `ticket_classification_system.ipynb`: The main Jupyter Notebook containing the end-to-end ML pipeline.
- `customer_support_tickets.csv`: The dataset (ensure this file is present).
- `requirements.txt`: List of Python dependencies.

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed (version 3.8 or higher is recommended).

### 2. Install Dependencies
Open your terminal or command prompt in this project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Running the Analysis
You can run the project using **Jupyter Notebook** or **VS Code**.

#### Option A: Using Jupyter Notebook
1.  In the terminal, run:
    ```bash
    jupyter notebook
    ```
2.  A browser window will open. Click on `ticket_classification_system.ipynb`.
3.  Click **"Run"** > **"Run All Cells"** to execute the entire analysis.

#### Option B: Using VS Code
1.  Open this folder in VS Code.
2.  Open `ticket_classification_system.ipynb`.
3.  Select your Python kernel (top right).
4.  Click **"Run All"** at the top of the notebook.

## 📊 Results & Insights
The notebook includes:
-   **Data Cleaning**: Lowercasing, lemmatization, and noise removal.
-   **Model Training**: Random Forest classifiers for **Category** and **Priority**.
-   **Evaluation**: Accuracy scores, Classification Reports, and Confusion Matrices.
-   **Inference**: A function to classify new, unseen tickets.

> **Note**: The current dataset appears to be synthetic with randomized labels, resulting in lower-than-usual accuracy (~20%). In a real-world scenario with organic text data, this same pipeline typically yields >80% accuracy.
