#!/usr/bin/env python3
"""
Basic ML Modelling for Loan Approval Classification
MSML Project - Kriteria 2 (Basic Level)

This script trains machine learning models using MLflow with autolog.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("Loan_Approval_Classification")

def load_preprocessed_data():
    """Load preprocessed data from kriteria 1"""
    try:
        X_train = pd.read_csv('loan_data_preprocessing/X_train.csv')
        X_test = pd.read_csv('loan_data_preprocessing/X_test.csv')
        y_train = pd.read_csv('loan_data_preprocessing/y_train.csv').iloc[:, 0]
        y_test = pd.read_csv('loan_data_preprocessing/y_test.csv').iloc[:, 0]
        
        print(f"Data loaded successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def train_model(X_train, X_test, y_train, y_test, model, model_name):
    """Train and evaluate a single model with MLflow autolog"""
    
    with mlflow.start_run(run_name=f"{model_name}_basic"):
        # Enable autolog for sklearn
        mlflow.sklearn.autolog()
        
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        # Create and log confusion matrix
        cm_plot = create_confusion_matrix_plot(y_test, y_pred_test, model_name)
        
        # Save plot
        cm_filename = f"confusion_matrix_{model_name.lower()}.png"
        cm_plot.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()
        
        # Log additional metrics (autolog will handle most of this)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        
        # Log model info
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_shape", f"{X_train.shape[0]}x{X_train.shape[1]}")
        
        print(f"✅ {model_name} training completed and logged to MLflow!")
        
        return {
            'model': model,
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }

def main():
    """Main training function"""
    print("🚀 Starting ML Model Training with MLflow")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        print("❌ Failed to load data. Please check data paths.")
        return
    
    # Define models to train
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        )
    }
    
    # Train all models
    results = []
    
    for model_name, model in models.items():
        try:
            result = train_model(X_train, X_test, y_train, y_test, model, model_name)
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Summary of results
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train Accuracy': f"{r['train_accuracy']:.4f}",
            'Test Accuracy': f"{r['test_accuracy']:.4f}",
            'Precision': f"{r['test_precision']:.4f}",
            'Recall': f"{r['test_recall']:.4f}",
            'F1-Score': f"{r['test_f1']:.4f}"
        }
        for r in results
    ])
    
    print(results_df.to_string(index=False))
    
    # Find best model
    if results:
        best_model = max(results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model['model_name']}")
        print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
        
        # Log best model info
        with mlflow.start_run(run_name="best_model_summary"):
            mlflow.log_param("best_model", best_model['model_name'])
            mlflow.log_metric("best_accuracy", best_model['test_accuracy'])
            
            # Save results summary
            results_df.to_csv("model_comparison.csv", index=False)
            mlflow.log_artifact("model_comparison.csv")
    
    print(f"\n✅ Training completed! Check MLflow UI at http://127.0.0.1:5000")
    print("   Run: mlflow ui --host 127.0.0.1 --port 5000")

if __name__ == "__main__":
    main()