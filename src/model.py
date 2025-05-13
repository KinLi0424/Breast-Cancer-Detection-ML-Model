from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, Y_train, learning_rate=0.0001, max_iter=1000):
    # Initialize and train the model
    model = LogisticRegression(C=1/learning_rate, max_iter=max_iter, solver='lbfgs')
    model.fit(X_train, Y_train)
    
    # Save the model
    joblib.dump(model, 'model.joblib')
    print("Model saved as model.joblib")
    return model

# Example usage:
# from data_preprocessing import load_and_preprocess_data
# X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
# train_model(X_train, Y_train)
