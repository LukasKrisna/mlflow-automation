import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np
import warnings
import sys
import joblib

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Parse command line arguments with optimized defaults
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    file_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    
    print(f"Training with optimized parameters:")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"min_samples_split: {min_samples_split}")
    print(f"min_samples_leaf: {min_samples_leaf}")

    # Read the dataset
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2,
        stratify=data["Credit_Score"]  # Ensure balanced splits
    )
    
    input_example = X_train.head(5)
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("dataset_shape", f"{data.shape[0]}x{data.shape[1]}")
        
        # Create optimized RandomForest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            warm_start=False,
            oob_score=True  # Out-of-bag score for additional validation
        )
        
        print("Training model...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        oob_score = model.oob_score_
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("oob_score", oob_score)
        mlflow.log_metric("model_size_mb", sys.getsizeof(model) / (1024 * 1024))
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        # Log model with optimized settings
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"OOB Score: {oob_score:.4f}")
        print(f"Model uses {n_estimators} estimators (reduced from 505 for efficiency)")
        
        # Print classification report for detailed analysis
        report = classification_report(y_test, y_pred, output_dict=True)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"{class_name}_precision", metrics.get('precision', 0))
                mlflow.log_metric(f"{class_name}_recall", metrics.get('recall', 0))
                mlflow.log_metric(f"{class_name}_f1", metrics.get('f1-score', 0))