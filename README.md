# 🧪 ETL Pipeline 

This repository contains a simple and modular **ETL (Extract, Transform, Load)** pipeline script built with Python.
The script reads a CSV file, processes it using standard imputation, encoding, and scaling techniques, and outputs clean data ready for machine learning or analysis.


## 📁 Structure
Task1/
├── etl_pipeline.py # Main ETL script
└── data.csv # Input dataset



---

## 🚀 Features

- Reads data from CSV
- Drops unnecessary columns
- Imputes missing values
- Encodes categorical features
- Scales numerical data
- Uses scikit-learn pipelines for clean transformations

---

## ⚙️ Setup

## bash
pip install pandas scikit-learn
python etl_pipeline.py


🔧 Main Functions
extract_data(): Loads and cleans the dataset.

build_pipeline(): Builds preprocessing steps.

run_pipeline(): Applies transformation and outputs processed data.

📌 Output
Console logs indicate each ETL step, ending with the final transformed dataset shape.


