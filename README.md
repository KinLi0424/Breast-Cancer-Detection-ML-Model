""# Breast Cancer Detection using Logistic Regression
This project implements a **Logistic Regression** model to classify breast cancer as benign or malignant, using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. 

---

## Project Overview
The aim of this project is to leverage logistic regression for medical diagnostics, focusing on precision and recall to minimize false negatives in identifying malignant cases. The project includes:
- Data Preprocessing
- Model Training and Hyperparameter Tuning
- Evaluation Metrics (Accuracy, Precision, Recall, F1 Score)
- Threshold Analysis for Optimal Decision Making

---

## Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features:** 30 numerical features representing characteristics of the cell nuclei.
- **Labels:** 
  - `0` → Benign
  - `1` → Malignant

---

## Dependencies
To install the required dependencies:
```bash
pip install -r requirements.txt
```

Dependencies include:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `jupyter`

---

## Usage
To run the notebook locally:
```bash
jupyter notebook Coursework2.ipynb
```
Open the notebook and run the cells sequentially to:
1. Load and preprocess the dataset.
2. Train the Logistic Regression model.
3. Evaluate the model with precision, recall, and F1 metrics.
4. Visualize the model's performance.

---

## Model Performance
Key metrics:
- **Accuracy:** 50%
- **Precision:** 100% (Perfect identification of positive predictions)
- **Recall:** 20% (Low detection of actual malignant cases)
- **F1 Score:** 33.33% (Indicates imbalance between precision and recall)

---

## Report
A detailed analysis of the model's performance and optimization is available in [Coursework2.pdf](./Coursework2.pdf).

---

## Contributions
If you want to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
""
