import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    # Load the model
    model = joblib.load('model.joblib')
    print("Model loaded successfully.")
    
    # Load data
    _, X_test, _, Y_test = load_and_preprocess_data()
    
    # Predict
    predictions = model.predict(X_test)
    
    # Metrics
    print(f"Accuracy: {accuracy_score(Y_test, predictions):.2f}")
    print(f"Precision: {precision_score(Y_test, predictions):.2f}")
    print(f"Recall: {recall_score(Y_test, predictions):.2f}")
    print(f"F1 Score: {f1_score(Y_test, predictions):.2f}")

# Example usage:
# evaluate_model()
