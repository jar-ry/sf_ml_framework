"""
Node 05: Train Model

This node trains an XGBoost model for customer churn prediction.
In local mode, reads/saves to data/ folder. In deployment mode, uses Snowflake Feature Store and Model Registry.
"""

import logging
import os
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Any, Tuple
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.utils.data_catalog import get_catalog

logger = logging.getLogger(__name__)


def train_model(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Train XGBoost model for customer churn prediction with cross-validation and hyperparameter optimization.
    
    Detects execution mode:
    - Local mode (feature_store=None): Reads from data/ folder, saves model locally
    - Deployment mode (feature_store provided): Uses Snowflake Feature Store and Model Registry
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (None for local execution)
        inputs: Dictionary containing:
            - train_feature_view: Name of the training feature view
            - version: Version of the training feature view
            - model_params: XGBoost hyperparameters (optional)
            - cv_folds: Number of cross-validation folds (default: 5)
            - early_stopping_rounds: Early stopping rounds (default: 50)
        outputs: Dictionary containing:
            - model_name: Name of the trained model
            - model_version: Version of the model
            - metrics_output: Whether to save training metrics (default: True)
    
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
        
        train_feature_view = inputs.get('train_feature_view', 'train_features')
        model_name = outputs.get('model_name', 'xgboost_churn_model')
        save_metrics = outputs.get('metrics_output', True)
        
        # Get versions from catalog instead of hardcoded defaults
        source_version = catalog.get_default_version(train_feature_view)
        model_version = catalog.get_default_version(model_name, 'models')
        
        # Model hyperparameters
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        model_params = inputs.get('model_params', default_params)
        
        # Training parameters
        cv_folds = inputs.get('cv_folds', 5)
        early_stopping_rounds = inputs.get('early_stopping_rounds', 50)
        
        logger.info(f"Training model: {model_name}:{model_version}")
        logger.info(f"Data source: {train_feature_view}:{source_version}")
        logger.info(f"Cross-validation folds: {cv_folds}")
        
        # Load training data based on execution mode
        if is_local_mode:
            # LOCAL MODE: Read from data/ folder
            logger.info("📁 Loading training data from local data/ folder...")
            
            # Try multiple possible file locations
            possible_files = [
                f"data/04_train_test_split/train_data_{train_feature_view}.csv",
                f"data/train_data_{train_feature_view}.csv",
                f"train_data_{train_feature_view}.csv",
                f"data/{train_feature_view}.csv"
            ]
            
            train_df = None
            for filepath in possible_files:
                if os.path.exists(filepath):
                    train_df = pd.read_csv(filepath)
                    logger.info(f"✅ Loaded training data from: {filepath}")
                    break
            
            if train_df is None:
                raise FileNotFoundError(f"Could not find training data. Tried: {possible_files}")
                
        else:
            # DEPLOYMENT MODE: Read from Snowflake Feature Store
            logger.info("☁️  Loading training data from Snowflake Feature Store...")
            
            # Retrieve training data from Feature Store
            feature_view = feature_store.get_feature_view(train_feature_view, source_version)
            train_df = feature_view.to_pandas()
            logger.info(f"✅ Loaded training data from Snowflake Feature Store: {train_feature_view}:{source_version}")
        
        logger.info(f"Loaded {len(train_df)} training records with {len(train_df.columns)} columns")
        
        # Prepare data for training
        logger.info("🔍 Preparing data for model training...")
        
        # Identify target and feature columns
        target_col = 'churned'
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        # Exclude non-feature columns
        exclude_cols = ['customer_id', target_col, 'split_type', 'split_timestamp']
        feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # Extract features and target
        X_train = train_df[feature_columns]
        y_train = train_df[target_col]
        
        logger.info(f"Training features: {len(feature_columns)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Churn rate: {y_train.mean():.1%}")
        
        # Data quality checks
        logger.info("🔍 Performing data quality checks...")
        
        # Check for missing values
        missing_values = X_train.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in features")
            # Fill with 0 for now (features should be preprocessed)
            X_train = X_train.fillna(0)
            logger.info("Filled missing values with 0")
        
        # Check for infinite values
        infinite_values = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            logger.warning(f"Found {infinite_values} infinite values in features")
            # Replace with large finite values
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            logger.info("Replaced infinite values with 0")
        
        # Log feature statistics
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
        logger.info(f"Feature types: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        
        # Cross-validation before final training
        logger.info("🔄 Performing cross-validation...")
        
        # Create base model for CV
        cv_model = xgb.XGBClassifier(
            **{k: v for k, v in model_params.items() if k != 'eval_metric'}
        )
        
        # Perform stratified cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate multiple metrics
        cv_scores = {}
        for metric_name, metric_func in [
            ('accuracy', 'accuracy'),
            ('precision', 'precision'),
            ('recall', 'recall'),
            ('f1', 'f1'),
            ('roc_auc', 'roc_auc')
        ]:
            scores = cross_val_score(cv_model, X_train, y_train, cv=cv_strategy, scoring=metric_func)
            cv_scores[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            logger.info(f"CV {metric_name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        # Train final model
        logger.info("🎯 Training final XGBoost model...")
        
        # Create final model
        model = xgb.XGBClassifier(**model_params)
        
        # Train the model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
        
        # Generate training predictions for metrics
        logger.info("📊 Evaluating model performance on training data...")
        
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        # Calculate training metrics
        training_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1_score': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        }
        
        logger.info("Training Performance:")
        for metric, value in training_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Feature importance analysis
        logger.info("🔍 Analyzing feature importance...")
        
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)
        logger.info("Top 10 most important features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Model metadata
        model_metadata = {
            'execution_mode': mode.lower(),
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version,
            'model_type': 'XGBClassifier',
            'training_data_source': f"{train_feature_view}:{source_version}",
            'training_samples': len(X_train),
            'feature_count': len(feature_columns),
            'target_column': target_col,
            'training_time_seconds': training_time,
            'hyperparameters': model_params,
            'cross_validation': {
                'folds': cv_folds,
                'scores': cv_scores
            },
            'training_metrics': training_metrics,
            'feature_columns': feature_columns,
            'top_features': top_features.to_dict('records')
        }
        
        # Save model and metadata based on execution mode
        if is_local_mode:
            # LOCAL MODE: Save to data/ folder
            logger.info("💾 Saving model and metadata to local data/ folder...")
            
            data_dir = "data/05_train_model/"
            os.makedirs(data_dir, exist_ok=True)
            
            # Save trained model
            model_filename = f"model_{model_name}_{model_version}.pkl"
            model_filepath = os.path.join(data_dir, model_filename)
            
            with open(model_filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Save feature importance
            importance_filename = f"feature_importance_{model_name}_{model_version}.csv"
            importance_filepath = os.path.join(data_dir, importance_filename)
            feature_importance.to_csv(importance_filepath, index=False)
            
            # Save model metadata
            metadata_filename = f"model_metadata_{model_name}_{model_version}.json"
            metadata_filepath = os.path.join(data_dir, metadata_filename)
            
            model_metadata['model_file_path'] = model_filepath
            model_metadata['importance_file_path'] = importance_filepath
            
            with open(metadata_filepath, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info(f"✅ Model saved to: {model_filepath}")
            logger.info(f"✅ Feature importance saved to: {importance_filepath}")
            logger.info(f"✅ Model metadata saved to: {metadata_filepath}")
            
        else:
            # DEPLOYMENT MODE: Save to Snowflake Model Registry
            logger.info("☁️  Saving model to Snowflake Model Registry...")
            
            # Register the model in Snowflake Model Registry
            model_registry.log_model(
                model_name=model_name,
                model_version=model_version,
                model_object=model,
                metadata=model_metadata,
                feature_columns=feature_columns
            )
            
            logger.info(f"✅ Model '{model_name}:{model_version}' registered in Snowflake Model Registry")
            logger.info(f"📊 Model metadata and feature importance logged successfully")
        
        # Training summary
        logger.info(f"🎯 Model training completed successfully in {mode} mode")
        logger.info(f"📊 Model Performance Summary:")
        logger.info(f"  - Training samples: {len(X_train)}")
        logger.info(f"  - Features: {len(feature_columns)}")
        logger.info(f"  - Training time: {training_time:.2f} seconds")
        logger.info(f"  - Accuracy: {training_metrics['accuracy']:.4f}")
        logger.info(f"  - F1-Score: {training_metrics['f1_score']:.4f}")
        logger.info(f"  - ROC-AUC: {training_metrics['roc_auc']:.4f}")
        logger.info(f"  - Top feature: {top_features.iloc[0]['feature']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to train model: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing train_model node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "train_feature_view": "test_train_features",
        "version": "v1",
        "cv_folds": 3,  # Reduced for testing
        "model_params": {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 50,  # Reduced for testing
            'random_state': 42
        }
    }
    
    test_outputs = {
        "model_name": "test_xgboost_churn_model",
        "model_version": "v1",
        "metrics_output": True
    }
    
    # Run in local mode (feature_store=None)
    train_model(
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
        def log_model(self, model_name, model_version, model_object, **kwargs):
            print(f"Mock: Would register model {model_name}:{model_version}")
    
    # Run in deployment mode (feature_store provided)
    train_model(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=MockModelRegistry(),
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 