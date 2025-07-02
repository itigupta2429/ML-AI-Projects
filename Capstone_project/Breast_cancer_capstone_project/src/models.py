import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_recall_curve, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def logistic_regression_classifier(X_train, X_test, y_train, y_test, 
                                 max_iter=1000, plot_curves=True, 
                                 tune_threshold=False, default_threshold=0.5,
                                 save_plots=True, plot_name_prefix="logreg"):
    """
    Enhanced Logistic Regression classifier with threshold tuning and plot saving.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        max_iter: Maximum iterations for LogisticRegression
        plot_curves: Whether to generate evaluation plots
        tune_threshold: Whether to find optimal threshold (default: False)
        default_threshold: Threshold to use if tune_threshold=False
        save_plots: Whether to save plots to results/plots/ (default: True)
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained LogisticRegression model
        metrics: Dictionary of evaluation metrics
        optimal_threshold: The optimal threshold found (if tune_threshold=True)
    """
    # Create plot directory if needed
    if save_plots or plot_curves:
        os.makedirs("results/plots", exist_ok=True)
    
    # Convert labels to {0, 1} if they're {2, 4}
    if set(y_train) == {2, 4}:
        y_train = y_train.replace({2: 0, 4: 1})
        y_test = y_test.replace({2: 0, 4: 1})
    
    # Train model
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Threshold tuning
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = default_threshold
    
    # Make predictions using selected threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold_used': optimal_threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Plots
    if plot_curves:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0, 0])
        ax[0, 0].set_title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
        
        # ROC Curve with optimal threshold marker
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0, 1])
        ax[0, 1].plot([0, 1], [0, 1], 'k--')
        if tune_threshold:
            ax[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], 
                            marker='o', color='red', 
                            label=f'Optimal Threshold ({optimal_threshold:.2f})')
            ax[0, 1].legend()
        ax[0, 1].set_title("ROC Curve")
        
        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1, 0])
        ax[1, 0].set_title("Precision-Recall Curve")
        
        # Calibration Curve
        CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10, ax=ax[1, 1])
        ax[1, 1].set_title("Calibration Curve")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved evaluation plots to {plot_path}")
        
        if plot_curves:
            plt.show()
        else:
            plt.close(fig)
    
    return model, metrics, (optimal_threshold if tune_threshold else default_threshold)

def SVM_classifier(X_train, X_test, y_train, y_test, 
                 kernel='linear', C=1.0, plot_curves=True,
                 tune_threshold=False, default_threshold=0.5,
                 save_plots=True, plot_name_prefix="svm"):
    """
    Enhanced SVM classifier with threshold tuning and plot saving.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        kernel: Kernel type for SVM (default: 'linear')
        C: Regularization parameter (default: 1.0)
        plot_curves: Whether to generate evaluation plots
        tune_threshold: Whether to find optimal threshold (default: False)
        default_threshold: Threshold to use if tune_threshold=False
        save_plots: Whether to save plots to results/plots/ (default: True)
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained SVC model
        metrics: Dictionary of evaluation metrics
        optimal_threshold: The optimal threshold found (if tune_threshold=True)
    """
    # Create plot directory if needed
    if save_plots or plot_curves:
        os.makedirs("results/plots", exist_ok=True)
    
    # Convert labels to {0, 1} if they're {2, 4}
    if set(y_train) == {2, 4}:
        y_train = y_train.replace({2: 0, 4: 1})
        y_test = y_test.replace({2: 0, 4: 1})
    
    # Train model (enable probability for ROC curves)
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predicted probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Threshold tuning
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = default_threshold
    
    # Make predictions using selected threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold_used': optimal_threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'kernel': kernel,
        'C': C
    }
    
    # Plots
    if plot_curves:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0, 0])
        ax[0, 0].set_title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
        
        # ROC Curve with optimal threshold marker
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0, 1])
        ax[0, 1].plot([0, 1], [0, 1], 'k--')
        if tune_threshold:
            ax[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], 
                            marker='o', color='red', 
                            label=f'Optimal Threshold ({optimal_threshold:.2f})')
            ax[0, 1].legend()
        ax[0, 1].set_title("ROC Curve")
        
        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1, 0])
        ax[1, 0].set_title("Precision-Recall Curve")
        
        # Calibration Curve
        CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10, ax=ax[1, 1])
        ax[1, 1].set_title("Calibration Curve")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_{kernel}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved SVM evaluation plots to {plot_path}")
        
        if plot_curves:
            plt.show()
        else:
            plt.close(fig)
    
    return model, metrics, optimal_threshold if tune_threshold else default_threshold

def decision_tree_classifier(X_train, X_test, y_train, y_test,
                           max_depth=None, min_samples_split=2,
                           plot_curves=True, tune_threshold=False,
                           default_threshold=0.5, save_plots=True,
                           plot_name_prefix="decision_tree"):
    """
    Enhanced Decision Tree classifier with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        plot_curves: Whether to generate evaluation plots
        tune_threshold: Whether to find optimal threshold
        default_threshold: Threshold if tune_threshold=False
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained DecisionTreeClassifier
        metrics: Dictionary of evaluation metrics
        optimal_threshold: Optimal threshold if tuned
    """
    os.makedirs("results/plots", exist_ok=True)
    
    # Label conversion
    if set(y_train) == {2, 4}:
        y_train = y_train.replace({2: 0, 4: 1})
        y_test = y_test.replace({2: 0, 4: 1})
    
    # Train model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get probabilities and tune threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = default_threshold
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold_used': optimal_threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    # Visualization
    if plot_curves:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0, 0])
        ax[0, 0].set_title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
        
        # ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0, 1])
        ax[0, 1].plot([0, 1], [0, 1], 'k--')
        if tune_threshold:
            ax[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], 
                             marker='o', color='red',
                             label=f'Optimal Threshold ({optimal_threshold:.2f})')
            ax[0, 1].legend()
        ax[0, 1].set_title("ROC Curve")
        
        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1, 0])
        ax[1, 0].set_title("Precision-Recall Curve")
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            pd.Series(model.feature_importances_, 
                      index=X_train.columns).sort_values().plot.barh(ax=ax[1, 1])
            ax[1, 1].set_title("Feature Importance")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Decision Tree evaluation plots to {plot_path}")
        
        plt.show() if plot_curves else plt.close(fig)
    
    return model, metrics, optimal_threshold if tune_threshold else default_threshold

def random_forest_classifier(X_train, X_test, y_train, y_test,
                            n_estimators=100, max_depth=None,
                            min_samples_split=2, plot_curves=True,
                            tune_threshold=False, default_threshold=0.5,
                            save_plots=True, plot_name_prefix="random_forest"):
    """
    Enhanced Random Forest classifier with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        plot_curves: Whether to generate evaluation plots
        tune_threshold: Whether to find optimal threshold
        default_threshold: Threshold if tune_threshold=False
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained RandomForestClassifier
        metrics: Dictionary of evaluation metrics
        optimal_threshold: Optimal threshold if tuned
    """
    os.makedirs("results/plots", exist_ok=True)
    
    # Label conversion
    if set(y_train) == {2, 4}:
        y_train = y_train.replace({2: 0, 4: 1})
        y_test = y_test.replace({2: 0, 4: 1})
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get probabilities and tune threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    
    if tune_threshold:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = default_threshold
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'threshold_used': optimal_threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    # Visualization
    if plot_curves:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0, 0])
        ax[0, 0].set_title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
        
        # ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0, 1])
        ax[0, 1].plot([0, 1], [0, 1], 'k--')
        if tune_threshold:
            ax[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], 
                             marker='o', color='red',
                             label=f'Optimal Threshold ({optimal_threshold:.2f})')
            ax[0, 1].legend()
        ax[0, 1].set_title("ROC Curve")
        
        # Precision-Recall Curve
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1, 0])
        ax[1, 0].set_title("Precision-Recall Curve")
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            pd.Series(model.feature_importances_, 
                      index=X_train.columns).sort_values().plot.barh(ax=ax[1, 1])
            ax[1, 1].set_title("Feature Importance")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Random Forest evaluation plots to {plot_path}")
        
        plt.show() if plot_curves else plt.close(fig)
    return model, metrics, optimal_threshold if tune_threshold else default_threshold

def neural_network_classifier(X_train, X_test, y_train, y_test,
                                epochs=50, batch_size=32,
                                plot_curves=True, tune_threshold=False,
                                default_threshold=0.5, save_plots=True,
                                plot_name_prefix="neural_network"):
        """
        Enhanced Neural Network classifier with evaluation metrics and plots.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            epochs: Number of training epochs
            batch_size: Size of training batches
            plot_curves: Whether to generate evaluation plots
            tune_threshold: Whether to find optimal threshold
            default_threshold: Threshold if tune_threshold=False
            save_plots: Whether to save plots
            plot_name_prefix: Prefix for saved plot filenames
            
        Returns:
            model: Trained Keras Sequential model
            metrics: Dictionary of evaluation metrics
            optimal_threshold: Optimal threshold if tuned
        """
        os.makedirs("results/plots", exist_ok=True)
        
        # Label conversion
        if set(y_train) == {2, 4}:
            y_train = y_train.replace({2: 0, 4: 1})
            y_test = y_test.replace({2: 0, 4: 1})
        
        # Build model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, verbose=0)
        
        # Get predicted probabilities
        y_proba = model.predict(X_test).flatten()
        
        # Threshold tuning
        if tune_threshold:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = default_threshold
        
        # Make predictions using selected threshold
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'threshold_used': optimal_threshold,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'epochs': epochs,
            'batch_size': batch_size
        }
        # Visualization
        if plot_curves:
            fig, ax = plt.subplots(2, 2, figsize=(15, 12))
            
            # Confusion Matrix
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0, 0])
            ax[0, 0].set_title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
            
            # ROC Curve
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[0, 1])
            ax[0, 1].plot([0, 1], [0, 1], 'k--')
            if tune_threshold:
                ax[0, 1].scatter(fpr[optimal_idx], tpr[optimal_idx], 
                                 marker='o', color='red',
                                 label=f'Optimal Threshold ({optimal_threshold:.2f})')
                ax[0, 1].legend()
            ax[0, 1].set_title("ROC Curve")
            
            # Precision-Recall Curve
            PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1, 0])
            ax[1, 0].set_title("Precision-Recall Curve")
            
            # Training History
            ax[1, 1].plot(history.history['accuracy'], label='Train Accuracy')
            ax[1, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
            ax[1, 1].set_title("Training History")
            ax[1, 1].set_xlabel("Epochs")
            ax[1, 1].set_ylabel("Accuracy")
            ax[1, 1].legend()
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved Neural Network evaluation plots to {plot_path}")
            
            plt.show() if plot_curves else plt.close(fig)
        return model, metrics, optimal_threshold if tune_threshold else default_threshold

# ==================== DATA PREPROCESSING FUNCTIONS FOR REGRESSION ====================

def preprocess_data(df, target_col='class', scale_features=True):
    """
    Preprocess the breast cancer dataset for machine learning tasks.
    
    Args:
        df: Input DataFrame
        target_col: Target column name ('class' for classification, 'bare_nucleoli' for regression)
        scale_features: Whether to scale features (default: True)
        
    Returns:
        X_train, X_test, y_train, y_test: Split and processed data
        scaler: Fitted scaler object (if scaling was performed)
    """
    # Convert 'class' to binary if it exists and is target
    if 'class' in df.columns and target_col == 'class':
        df['class'] = df['class'].map({2: 0, 4: 1})
        df = df.dropna(subset=['class'])
        df['class'] = df['class'].astype(int)
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    
    # Features to scale (exclude id, bare_nucleoli if not target, and target column)
    features_to_scale = [col for col in df.columns 
                        if col not in [target_col, 'id'] and 
                        not (col == 'bare_nucleoli' and target_col != 'bare_nucleoli')]
    
    if scale_features:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])
        
        # Create new DataFrame with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale, index=df.index)
        
        # Add back non-scaled columns
        if 'bare_nucleoli' in df.columns and target_col != 'bare_nucleoli':
            scaled_df['bare_nucleoli'] = df['bare_nucleoli'].values
        if 'id' in df.columns:
            scaled_df['id'] = df['id'].values
        
        df = scaled_df
    
    # Split into features and target
    X = df.drop(columns=[target_col, 'id'] if 'id' in df.columns else target_col)
    y = df[target_col]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, (scaler if scale_features else None)

# ==================== COMMON UTILITY FUNCTIONS ====================

def _prepare_directory(save_plots, plot_curves):
    """Create plot directory if needed"""
    if save_plots or plot_curves:
        os.makedirs("results/plots", exist_ok=True)

def _convert_labels(y_train, y_test):
    """Convert {2,4} labels to {0,1} if needed"""
    if set(y_train) == {2, 4}:
        y_train = y_train.replace({2: 0, 4: 1})
        y_test = y_test.replace({2: 0, 4: 1})
    return y_train, y_test


# ==================== REGRESSION MODELS ====================

def linear_regression(X_train, X_test, y_train, y_test, 
                     plot_results=True, save_plots=True,
                     plot_name_prefix="linear_regression"):
    """
    Linear Regression model with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        plot_results: Whether to generate evaluation plots
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained LinearRegression model
        metrics: Dictionary of evaluation metrics
    """
    _prepare_directory(save_plots, plot_results)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }
    
    # Visualization
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax[0].scatter(y_test, y_pred, alpha=0.5)
        ax[0].plot([y_test.min(), y_test.max()], 
                  [y_test.min(), y_test.max()], 'k--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title('Actual vs Predicted Values')
        
        # Residual Plot
        residuals = y_test - y_pred
        ax[1].scatter(y_pred, residuals, alpha=0.5)
        ax[1].axhline(y=0, color='k', linestyle='--')
        ax[1].set_xlabel('Predicted Values')
        ax[1].set_ylabel('Residuals')
        ax[1].set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Linear Regression evaluation plots to {plot_path}")
        
        plt.show() if plot_results else plt.close(fig)
    
    return model, metrics

def SVM_regressor(X_train, X_test, y_train, y_test, 
                 kernel='rbf', C=1.0, plot_results=True,
                 save_plots=True, plot_name_prefix="svm_regressor"):
    """
    SVM Regressor with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        kernel: Kernel type for SVM (default: 'rbf')
        C: Regularization parameter (default: 1.0)
        plot_results: Whether to generate evaluation plots
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained SVR model
        metrics: Dictionary of evaluation metrics
    """
    _prepare_directory(save_plots, plot_results)
    
    # Train model
    model = SVR(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'kernel': kernel,
        'C': C
    }
    
    # Visualization
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax[0].scatter(y_test, y_pred, alpha=0.5)
        ax[0].plot([y_test.min(), y_test.max()], 
                  [y_test.min(), y_test.max()], 'k--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title('Actual vs Predicted Values')
        
        # Residual Plot
        residuals = y_test - y_pred
        ax[1].scatter(y_pred, residuals, alpha=0.5)
        ax[1].axhline(y=0, color='k', linestyle='--')
        ax[1].set_xlabel('Predicted Values')
        ax[1].set_ylabel('Residuals')
        ax[1].set_title('Residual Plot')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_{kernel}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved SVM Regressor evaluation plots to {plot_path}")
        
        plt.show() if plot_results else plt.close(fig)
    
    return model, metrics

def decision_tree_regressor(X_train, X_test, y_train, y_test,
                          max_depth=None, min_samples_split=2,
                          plot_results=True, save_plots=True,
                          plot_name_prefix="decision_tree_regressor"):
    """
    Decision Tree regressor with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        plot_results: Whether to generate evaluation plots
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained DecisionTreeRegressor
        metrics: Dictionary of evaluation metrics
    """
    _prepare_directory(save_plots, plot_results)
    
    # Train model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    # Visualization
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax[0].scatter(y_test, y_pred, alpha=0.5)
        ax[0].plot([y_test.min(), y_test.max()], 
                  [y_test.min(), y_test.max()], 'k--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title('Actual vs Predicted Values')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            pd.Series(model.feature_importances_, 
                     index=X_train.columns).sort_values().plot.barh(ax=ax[1])
            ax[1].set_title("Feature Importance")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Decision Tree Regressor evaluation plots to {plot_path}")
        
        plt.show() if plot_results else plt.close(fig)
    
    return model, metrics

def random_forest_regressor(X_train, X_test, y_train, y_test,
                          n_estimators=100, max_depth=None,
                          min_samples_split=2, plot_results=True,
                          save_plots=True, plot_name_prefix="random_forest_regressor"):
    """
    Random Forest regressor with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        plot_results: Whether to generate evaluation plots
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained RandomForestRegressor
        metrics: Dictionary of evaluation metrics
    """
    _prepare_directory(save_plots, plot_results)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    # Visualization
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax[0].scatter(y_test, y_pred, alpha=0.5)
        ax[0].plot([y_test.min(), y_test.max()], 
                  [y_test.min(), y_test.max()], 'k--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title('Actual vs Predicted Values')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            pd.Series(model.feature_importances_, 
                     index=X_train.columns).sort_values().plot.barh(ax=ax[1])
            ax[1].set_title("Feature Importance")
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Random Forest Regressor evaluation plots to {plot_path}")
        
        plt.show() if plot_results else plt.close(fig)
    
    return model, metrics

def neural_network_regressor(X_train, X_test, y_train, y_test,
                           epochs=50, batch_size=32, plot_results=True,
                           save_plots=True, plot_name_prefix="neural_network_regressor"):
    """
    Neural Network regressor with evaluation metrics and plots.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        epochs: Number of training epochs
        batch_size: Size of training batches
        plot_results: Whether to generate evaluation plots
        save_plots: Whether to save plots
        plot_name_prefix: Prefix for saved plot filenames
        
    Returns:
        model: Trained Keras Sequential model
        metrics: Dictionary of evaluation metrics
    """
    _prepare_directory(save_plots, plot_results)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    # Visualization
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax[0].scatter(y_test, y_pred, alpha=0.5)
        ax[0].plot([y_test.min(), y_test.max()], 
                  [y_test.min(), y_test.max()], 'k--')
        ax[0].set_xlabel('Actual')
        ax[0].set_ylabel('Predicted')
        ax[0].set_title('Actual vs Predicted Values')
        
        # Training History
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Val Loss')
        ax[1].set_title("Training History")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss (MSE)")
        ax[1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = f"results/plots/{plot_name_prefix}_eval_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved Neural Network Regressor evaluation plots to {plot_path}")
        
        plt.show() if plot_results else plt.close(fig)
    
    return model, metrics


