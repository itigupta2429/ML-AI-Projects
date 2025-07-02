#!/usr/bin/env python3
"""
Main execution script for Breast Cancer Classification and Regression Project
"""

import os
import sys
import inspect

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your project modules
try:
    from src.dataPreprocess_and_FeatureEng_ import (
        download_data,
        load_data,
        clean_data,
        scale_features,
        split_data,
        save_processed_data
    )
    from src.visualize import (
        plot_feature_distributions, 
        plot_correlation_heatmap,
        plot_feature_relationships
    )
    from src.models import (
        logistic_regression_classifier,
        SVM_classifier,
        decision_tree_classifier,
        random_forest_classifier,
        neural_network_classifier,
        linear_regression,
        SVM_regressor,
        decision_tree_regressor,
        random_forest_regressor,
        neural_network_regressor
    )
    from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
except ModuleNotFoundError as e:
    from config import RAW_DATA_PATH
    sys.exit(1)

def run_eda(df, target_col='class'):
    """Run Exploratory Data Analysis visualizations"""
    print(f"\n=== Running EDA Visualizations (target: {target_col}) ===")
    
    # Plot feature distributions (excluding ID and target columns)
    exclude_cols = ['id', target_col]
    plot_feature_distributions(df, exclude_cols=exclude_cols)
    
    # Ensure target is 1-dimensional for correlation heatmap
    target_series = df[target_col].squeeze() if target_col in df.columns else None
    if target_series is not None:
        # Plot correlation heatmap and get top features
        top_features = plot_correlation_heatmap(df, target_col=target_col)
        
        # Plot feature relationships if we found correlated features
        if top_features:
            plot_feature_relationships(df, top_features, target_col=target_col)
        
        return top_features
    return None
def run_pipeline(target_col='class'):
    """Run the complete data processing pipeline"""
    try:
        # Step 1: Download data (only needed once)
        if not os.path.exists(RAW_DATA_PATH):
            print("Downloading data...")
            download_data()
        
        # Step 2: Load and inspect data
        print("\n=== Loading Data ===")
        df = load_data()
        
        # Step 3: Clean data
        print("\n=== Cleaning Data ===")
        df = clean_data(df)
        
        # --- FIX 1: Always map 'class' to 0/1 after cleaning ---
        # This ensures 'class' is a proper binary feature for the regression task,
        # and a proper target for the classification task.
        if 'class' in df.columns:
            df['class'] = df['class'].map({2: 0, 4: 1}).astype(int)
        
        # Step 4: Run EDA visualizations
        top_features = run_eda(df, target_col=target_col)
        
        # --- FIX 2: Pass target_col to the updated scale_features function ---
        print("\n=== Scaling Features ===")
        df, scaler = scale_features(df, target_col)
        
        # Step 6: Split data
        print("\n=== Splitting Data ===")
        X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

        # --- FIX 3: Removed the np.ravel() bandage as it's no longer needed ---

        print("\n=== Saving Processed Data ===")
        save_processed_data(df)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'dataframe': df,
            'top_features': top_features,
            'target_col': target_col
        }
    
    except Exception as e:
        print(f"\nError in pipeline: {str(e)}", file=sys.stderr)
        raise

def run_classification_models(X_train, X_test, y_train, y_test):
    """Run all classification models"""
    models = {
        'Logistic Regression': logistic_regression_classifier,
        'SVM': SVM_classifier,
        'Decision Tree': decision_tree_classifier,
        'Random Forest': random_forest_classifier,
        'Neural Network': neural_network_classifier
    }
    return _run_models(models, X_train, X_test, y_train, y_test)

def run_regression_models(X_train, X_test, y_train, y_test):
    """Run all regression models"""
    models = {
        'Linear Regression': linear_regression,
        'SVM Regressor': SVM_regressor,
        'Decision Tree Regressor': decision_tree_regressor,
        'Random Forest Regressor': random_forest_regressor,
        'Neural Network Regressor': neural_network_regressor
    }
    return _run_models(models, X_train, X_test, y_train, y_test)

def _run_models(models, X_train, X_test, y_train, y_test):
    """Generic model runner that handles both classification and regression"""
    results = {}
    
    for name, model_func in models.items():
        print(f"\n=== Training {name} ===")
        try:
            # Get function parameters
            params = inspect.signature(model_func).parameters
            
            # Prepare common arguments
            kwargs = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'plot_curves': True
            }
            
            # Add neural network specific params if they exist
            if 'Neural Network' in name:
                kwargs.update({'epochs': 50, 'batch_size': 32})
            
            # Filter kwargs to only include parameters the function accepts
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
            
            # Execute model
            result = model_func(**filtered_kwargs)
            
            # Handle return values
            if len(result) == 3:
                model, metrics, threshold = result
            elif len(result) == 2:
                model, metrics = result
                threshold = None
            else:
                raise ValueError(f"Unexpected number of return values from {name}")
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'threshold': threshold
            }
            
            print(f"{name} completed successfully")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}", file=sys.stderr)
            results[name] = {'error': str(e)}
    
    return results

def print_results_summary(results, task_type='classification'):
    """Print formatted results summary"""
    print(f"\n=== {task_type.capitalize()} Model Results Summary ===")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: Failed - {result['error']}")
        else:
            metrics = result['metrics']
            if task_type == 'classification':
                print(f"{name}: Accuracy: {metrics['accuracy']:.4f}, "
                      f"Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, "
                      f"F1: {metrics['f1_score']:.4f}")
            else:  # regression
                print(f"{name}: R²: {metrics['r2']:.4f}, "
                      f"MSE: {metrics['mse']:.4f}, "
                      f"RMSE: {metrics['rmse']:.4f}, "
                      f"MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    try:
        print("Starting data processing pipeline...")
        
        # Run classification pipeline
        print("\n=== Running Classification Task ===")
        class_data = run_pipeline(target_col='class')
        class_results = run_classification_models(
            class_data['X_train'], 
            class_data['X_test'], 
            class_data['y_train'], 
            class_data['y_test']
        )
        print_results_summary(class_results, 'classification')
        
        # Run regression pipeline
        print("\n=== Running Regression Task ===")
        reg_data = run_pipeline(target_col='bare_nucleoli')
        reg_results = run_regression_models(
            reg_data['X_train'], 
            reg_data['X_test'], 
            reg_data['y_train'], 
            reg_data['y_test']
        )
        print_results_summary(reg_results, 'regression')
        
        print("\n=== All tasks completed successfully ===")
        
    except Exception as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        sys.exit(1)