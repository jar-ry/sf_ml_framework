"""
Node 06: Evaluate Model

This node evaluates the trained model on test data with comprehensive metrics and visualizations.
In local mode, reads/saves to data/ folder. In deployment mode, uses Snowflake Feature Store and Model Registry.
"""

import logging
import os
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import shap
from src.data_catalog import get_catalog

logger = logging.getLogger(__name__)

# Set matplotlib to use non-interactive backend for server environments
plt.switch_backend('Agg')


def evaluate_model(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Evaluate trained model on test data with comprehensive metrics and SHAP explanations.
    
    Detects execution mode:
    - Local mode (feature_store=None): Reads from data/ folder, saves results locally
    - Deployment mode (feature_store provided): Uses Snowflake Feature Store and Model Registry
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (None for local execution)
        inputs: Dictionary containing:
            - test_feature_view: Name of the test feature view
            - model_name: Name of the trained model to evaluate
            - model_version: Version of the model
            - version: Version of the test data
            - generate_shap: Whether to generate SHAP explanations (default: True)
            - save_plots: Whether to save visualization plots (default: True)
        outputs: Dictionary containing:
            - evaluation_results: Name for evaluation results output
            - version: Version of the evaluation results
    
    Returns:
        None
    """
    
    # Detect execution mode
    is_local_mode = feature_store is None
    mode = "LOCAL" if is_local_mode else "DEPLOYMENT"
    logger.info(f"🔄 Running in {mode} mode")
    
    try:
        # Extract parameters and get catalog for version resolution
        catalog = get_catalog()
        
        test_feature_view = inputs.get('test_feature_view', 'test_features')
        model_name = inputs.get('model_name', 'xgboost_churn_model')
        generate_shap = inputs.get('generate_shap', True)
        save_plots = inputs.get('save_plots', True)
        
        evaluation_results = outputs.get('evaluation_results', 'model_evaluation')
        
        # Get versions from catalog instead of hardcoded defaults
        model_version = catalog.get_default_version(model_name, 'models')
        data_version = catalog.get_default_version(test_feature_view)
        output_version = catalog.get_default_version(evaluation_results, 'evaluations')
        
        logger.info(f"Evaluating model: {model_name}:{model_version}")
        logger.info(f"Test data: {test_feature_view}:{data_version}")
        logger.info(f"SHAP analysis: {generate_shap}, Save plots: {save_plots}")
        
        # Load test data and model based on execution mode
        if is_local_mode:
            # LOCAL MODE: Read from data/ folder
            logger.info("📁 Loading test data and model from local data/ folder...")
            
            # Load test data
            test_files = [
                f"data/04_train_test_split/test_data_{test_feature_view}.csv",
                f"data/test_data_{test_feature_view}.csv",
                f"test_data_{test_feature_view}.csv",
                f"data/{test_feature_view}.csv"
            ]
            
            test_df = None
            for filepath in test_files:
                if os.path.exists(filepath):
                    test_df = pd.read_csv(filepath)
                    logger.info(f"✅ Loaded test data from: {filepath}")
                    break
            
            if test_df is None:
                raise FileNotFoundError(f"Could not find test data. Tried: {test_files}")
            
            # Load trained model
            model_files = [
                f"data/05_train_model/model_{model_name}_{model_version}.pkl",
                f"data/model_{model_name}_{model_version}.pkl",
                f"model_{model_name}_{model_version}.pkl"
            ]
            
            model = None
            for filepath in model_files:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"✅ Loaded model from: {filepath}")
                    break
            
            if model is None:
                raise FileNotFoundError(f"Could not find trained model. Tried: {model_files}")
                
        else:
            # DEPLOYMENT MODE: Load from Snowflake
            logger.info("☁️  Loading test data and model from Snowflake...")
            
            # Load test data from Feature Store
            feature_view = feature_store.get_feature_view(test_feature_view, data_version)
            test_df = feature_view.to_pandas()
            logger.info(f"✅ Loaded test data from Snowflake Feature Store: {test_feature_view}:{data_version}")
            
            # Load model from Model Registry
            model = model_registry.load_model(model_name, model_version)
            logger.info(f"✅ Loaded model from Snowflake Model Registry: {model_name}:{model_version}")
        
        logger.info(f"Loaded {len(test_df)} test records")
        
        # Prepare test data for evaluation
        logger.info("🔍 Preparing test data for evaluation...")
        
        # Identify target and feature columns
        target_col = 'churned'
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Exclude non-feature columns
        exclude_cols = ['customer_id', target_col, 'split_type', 'split_timestamp']
        feature_columns = [col for col in test_df.columns if col not in exclude_cols]
        
        # Extract features and target
        X_test = test_df[feature_columns]
        y_test = test_df[target_col]
        
        logger.info(f"Test features: {len(feature_columns)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        logger.info(f"Test churn rate: {y_test.mean():.1%}")
        
        # Handle missing values (should be minimal after preprocessing)
        missing_values = X_test.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in test features")
            X_test = X_test.fillna(0)
            logger.info("Filled missing values with 0")
        
        # Generate predictions
        logger.info("🎯 Generating model predictions...")
        
        start_time = datetime.now()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prediction_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"✅ Generated predictions in {prediction_time:.4f} seconds")
        logger.info(f"Predicted churn rate: {y_pred.mean():.1%}")
        
        # Calculate comprehensive metrics
        logger.info("📊 Calculating performance metrics...")
        
        # Basic classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Performance metrics dictionary
        performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'classification_report': class_report,
            'prediction_time_seconds': prediction_time
        }
        
        # Log key metrics
        logger.info("🎯 Model Performance on Test Data:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Create visualizations if requested
        plots_saved = []
        if save_plots:
            logger.info("📈 Creating visualization plots...")
            
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Set plot style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix Heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Churn', 'Churn'],
                       yticklabels=['Not Churn', 'Churn'])
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            cm_plot_path = os.path.join(data_dir, f"confusion_matrix_{model_name}_{model_version}.png")
            plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_saved.append(cm_plot_path)
            logger.info(f"  Saved confusion matrix: {cm_plot_path}")
            
            # 2. ROC Curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.05))
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            roc_plot_path = os.path.join(data_dir, f"roc_curve_{model_name}_{model_version}.png")
            plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_saved.append(roc_plot_path)
            logger.info(f"  Saved ROC curve: {roc_plot_path}")
            
            # 3. Precision-Recall Curve
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_curve, precision_curve, color='blue', lw=2, 
                   label=f'PR curve (F1 = {f1:.3f})')
            baseline_precision = float(y_test.mean())
            ax.axhline(y=baseline_precision, color='red', linestyle='--', 
                      label=f'Baseline ({baseline_precision:.3f})')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.05))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            pr_plot_path = os.path.join(data_dir, f"precision_recall_{model_name}_{model_version}.png")
            plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_saved.append(pr_plot_path)
            logger.info(f"  Saved precision-recall curve: {pr_plot_path}")
            
            # 4. Prediction Distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Distribution of prediction probabilities
            ax1.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Not Churn', color='blue')
            ax1.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Churn', color='red')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Predicted Probabilities')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Prediction vs actual scatter
            scatter_x = np.random.normal(y_test, 0.1)  # Add jitter for visualization
            ax2.scatter(scatter_x, y_pred_proba, alpha=0.6, s=20)
            ax2.set_xlabel('Actual Class (with jitter)')
            ax2.set_ylabel('Predicted Probability')
            ax2.set_title('Predictions vs Actual')
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['Not Churn', 'Churn'])
            ax2.grid(True, alpha=0.3)
            
            pred_dist_path = os.path.join(data_dir, f"prediction_distribution_{model_name}_{model_version}.png")
            plt.savefig(pred_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_saved.append(pred_dist_path)
            logger.info(f"  Saved prediction distribution: {pred_dist_path}")
        
        # SHAP analysis if requested
        shap_results = None
        if generate_shap:
            logger.info("🔍 Generating SHAP explanations...")
            
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values for a sample of test data (to avoid memory issues)
                sample_size = min(100, len(X_test))
                sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample = X_test.iloc[sample_indices]
                
                shap_values = explainer.shap_values(X_sample)
                
                # SHAP summary statistics
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                feature_importance_shap = pd.DataFrame({
                    'feature': feature_columns,
                    'shap_importance': mean_abs_shap
                }).sort_values('shap_importance', ascending=False)
                
                shap_results = {
                    'sample_size': sample_size,
                    'feature_importance': feature_importance_shap.to_dict('records'),
                    'top_features': feature_importance_shap.head(10).to_dict('records')
                }
                
                logger.info("Top 10 features by SHAP importance:")
                for idx, row in feature_importance_shap.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['shap_importance']:.4f}")
                
                # Save SHAP plots if requested
                if save_plots:
                    # SHAP summary plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, 
                                    show=False, max_display=15)
                    shap_summary_path = os.path.join(data_dir, f"shap_summary_{model_name}_{model_version}.png")
                    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots_saved.append(shap_summary_path)
                    logger.info(f"  Saved SHAP summary plot: {shap_summary_path}")
                
                logger.info("✅ SHAP analysis completed")
                
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {str(e)}")
                shap_results = {"error": str(e)}
        
        # Compile evaluation results
        evaluation_data = {
            'execution_mode': mode.lower(),
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version,
            'test_data_source': f"{test_feature_view}:{data_version}",
            'test_samples': len(X_test),
            'feature_count': len(feature_columns),
            'performance_metrics': performance_metrics,
            'shap_analysis': shap_results,
            'plots_generated': plots_saved if save_plots else [],
            'feature_columns': feature_columns
        }
        
        # Save evaluation results based on execution mode
        if is_local_mode:
            # LOCAL MODE: Save to data/ folder
            logger.info("💾 Saving evaluation results to local data/ folder...")
            
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Save detailed evaluation results
            results_filename = f"evaluation_results_{evaluation_results}_{output_version}.json"
            results_filepath = os.path.join(data_dir, results_filename)
            
            with open(results_filepath, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            
            # Save predictions for further analysis
            predictions_df = test_df[['customer_id']].copy() if 'customer_id' in test_df.columns else pd.DataFrame()
            predictions_df['actual_churn'] = y_test
            predictions_df['predicted_churn'] = y_pred
            predictions_df['churn_probability'] = y_pred_proba
            predictions_df['correct_prediction'] = (y_test == y_pred)
            
            predictions_filename = f"predictions_{model_name}_{model_version}.csv"
            predictions_filepath = os.path.join(data_dir, predictions_filename)
            predictions_df.to_csv(predictions_filepath, index=False)
            
            logger.info(f"✅ Evaluation results saved to: {results_filepath}")
            logger.info(f"✅ Predictions saved to: {predictions_filepath}")
            if plots_saved:
                logger.info(f"✅ {len(plots_saved)} plots saved to data/ folder")
            
        else:
            # DEPLOYMENT MODE: Log to Snowflake
            logger.info("☁️  Logging evaluation results to Snowflake...")
            
            # Log evaluation metrics to Model Registry
            model_registry.log_metrics(
                model_name=model_name,
                model_version=model_version,
                metrics=performance_metrics,
                evaluation_data=evaluation_data
            )
            
            logger.info(f"✅ Evaluation results logged to Snowflake Model Registry")
            logger.info(f"📊 Performance metrics and analysis results stored successfully")
        
        # Final evaluation summary
        logger.info(f"🎯 Model evaluation completed successfully in {mode} mode")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"  - Test samples: {len(X_test)}")
        logger.info(f"  - Accuracy: {accuracy:.4f}")
        logger.info(f"  - Precision: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1-Score: {f1:.4f}")
        logger.info(f"  - ROC-AUC: {roc_auc:.4f}")
        if shap_results and 'top_features' in shap_results and len(shap_results['top_features']) > 0:
            top_features = shap_results['top_features']
            if isinstance(top_features, list) and len(top_features) > 0:
                first_feature = top_features[0]
                if isinstance(first_feature, dict) and 'feature' in first_feature:
                    top_shap_feature = first_feature['feature']
                    logger.info(f"  - Most important feature (SHAP): {top_shap_feature}")
        
    except Exception as e:
        logger.error(f"❌ Failed to evaluate model: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing evaluate_model node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "test_feature_view": "test_test_features",
        "model_name": "test_xgboost_churn_model",
        "model_version": "v1",
        "version": "v1",
        "generate_shap": True,
        "save_plots": True
    }
    
    test_outputs = {
        "evaluation_results": "test_model_evaluation",
        "version": "v1"
    }
    
    # Run in local mode (feature_store=None)
    evaluate_model(
        feature_store=None,  # Local mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    # Test deployment mode simulation
    print("\n=== Testing DEPLOYMENT mode simulation ===")
    
    # Create mock objects
    class MockFeatureStore:
        def get_feature_view(self, name, version):
            print(f"Mock: Would get feature view {name}:{version}")
            return None
    
    class MockModelRegistry:
        def load_model(self, model_name, model_version):
            print(f"Mock: Would load model {model_name}:{model_version}")
            return None
        
        def log_metrics(self, model_name, model_version, **kwargs):
            print(f"Mock: Would log metrics for model {model_name}:{model_version}")
    
    # Run in deployment mode (feature_store provided)
    evaluate_model(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=MockModelRegistry(),
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 