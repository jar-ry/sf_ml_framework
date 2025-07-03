"""
Node 04: Train/Test Split

This node performs intelligent data splitting for model training and validation.
In local mode, reads/saves to data/ folder. In deployment mode, uses Snowflake Feature Store.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.utils.data_catalog import get_catalog

logger = logging.getLogger(__name__)


def create_train_test_split(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Split engineered features into training and testing sets with quality validation.
    
    Detects execution mode:
    - Local mode (feature_store=None): Reads from data/ folder, saves to data/ folder
    - Deployment mode (feature_store provided): Uses Snowflake Feature Store
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (not used in this node)
        inputs: Dictionary containing:
            - source_feature_view: Name of the engineered feature view
            - version: Version of the source feature view
            - test_size: Proportion of data for testing (default: 0.2)
            - random_state: Random seed for reproducibility (default: 42)
            - stratify: Whether to stratify split by target (default: True)
        outputs: Dictionary containing:
            - train_feature_view: Name of the training feature view
            - test_feature_view: Name of the testing feature view
            - version: Version of the output feature views
    
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
        
        source_feature_view = inputs.get('source_feature_view', 'engineered_customer_features')
        test_size = inputs.get('test_size', 0.2)
        random_state = inputs.get('random_state', 42)
        stratify = inputs.get('stratify', True)
        
        train_feature_view = outputs.get('train_feature_view', 'train_features')
        test_feature_view = outputs.get('test_feature_view', 'test_features')
        
        # Get versions from catalog instead of hardcoded defaults
        source_version = catalog.get_default_version(source_feature_view)
        output_version = catalog.get_default_version(train_feature_view)  # Use train version as output version
        
        logger.info(f"Splitting data: {source_feature_view}:{source_version}")
        logger.info(f"→ {train_feature_view}:{output_version} & {test_feature_view}:{output_version}")
        logger.info(f"Test size: {test_size:.1%}, Stratify: {stratify}")
        
        # Load data based on execution mode
        if is_local_mode:
            # LOCAL MODE: Read from data/ folder
            logger.info("📁 Loading data from local data/ folder...")
            
            # Try multiple possible file locations
            possible_files = [
                f"data/03_feature_engineering/engineered_features_{source_feature_view}.csv",
                f"data/engineered_features_{source_feature_view}.csv",
                f"engineered_features_{source_feature_view}.csv",
                f"data/{source_feature_view}.csv"
            ]
            
            df = None
            for filepath in possible_files:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    logger.info(f"✅ Loaded data from: {filepath}")
                    break
            
            if df is None:
                raise FileNotFoundError(f"Could not find source data. Tried: {possible_files}")
                
        else:
            # DEPLOYMENT MODE: Read from Snowflake Feature Store
            logger.info("☁️  Loading data from Snowflake Feature Store...")
            
            # Retrieve data from Feature Store
            feature_view = feature_store.get_feature_view(source_feature_view, source_version)
            df = feature_view.to_pandas()
            logger.info(f"✅ Loaded data from Snowflake Feature Store: {source_feature_view}:{source_version}")
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Data quality checks before splitting
        logger.info("🔍 Performing pre-split data quality checks...")
        
        # Check for target variable
        target_col = 'churned'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Check target distribution
        target_counts = df[target_col].value_counts()
        target_ratio = df[target_col].mean()
        logger.info(f"Target distribution:")
        logger.info(f"  Churned (1): {target_counts.get(1, 0)} ({target_ratio:.1%})")
        logger.info(f"  Not Churned (0): {target_counts.get(0, 0)} ({1-target_ratio:.1%})")
        
        # Check for minimum class size
        min_class_size = min(target_counts)
        required_min_size = int(len(df) * test_size * 0.5)  # At least half of test size for minority class
        
        if min_class_size < required_min_size:
            logger.warning(f"Minority class size ({min_class_size}) is small relative to test size")
            logger.warning("Consider adjusting test_size or gathering more data")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values - this may affect split quality")
        
        # Identify feature types
        feature_columns = [col for col in df.columns if col not in ['customer_id', target_col]]
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[feature_columns].select_dtypes(include=['object', 'bool']).columns.tolist()
        
        logger.info(f"Features for splitting:")
        logger.info(f"  Total features: {len(feature_columns)}")
        logger.info(f"  Numeric features: {len(numeric_features)}")
        logger.info(f"  Categorical features: {len(categorical_features)}")
        
        # Perform the split
        logger.info("✂️  Performing train/test split...")
        
        X = df[feature_columns]
        y = df[target_col]
        
        # Additional columns to preserve
        additional_cols = []
        if 'customer_id' in df.columns:
            additional_cols.append('customer_id')
        
        stratify_y = y if stratify else None
        
        try:
            if additional_cols:
                # Split with additional columns preserved
                train_indices, test_indices = train_test_split(
                    range(len(df)),
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_y
                )
                
                train_df = df.iloc[train_indices].copy()
                test_df = df.iloc[test_indices].copy()
            else:
                # Standard split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_y
                )
                
                # Reconstruct DataFrames properly
                train_df = X_train.copy()
                train_df[target_col] = y_train
                test_df = X_test.copy()
                test_df[target_col] = y_test
            
            logger.info(f"✅ Split completed successfully")
            logger.info(f"  Training set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
            logger.info(f"  Test set: {len(test_df)} records ({len(test_df)/len(df):.1%})")
            
        except ValueError as e:
            if "stratify" in str(e).lower():
                logger.warning("Stratification failed, falling back to random split")
                stratify_y = None
                
                if additional_cols:
                    train_indices, test_indices = train_test_split(
                        range(len(df)),
                        test_size=test_size,
                        random_state=random_state,
                        stratify=stratify_y
                    )
                    
                    train_df = df.iloc[train_indices].copy()
                    test_df = df.iloc[test_indices].copy()
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=stratify_y
                    )
                    
                    # Reconstruct DataFrames properly
                    train_df = X_train.copy()
                    train_df[target_col] = y_train
                    test_df = X_test.copy()
                    test_df[target_col] = y_test
                
                logger.info(f"✅ Random split completed")
                logger.info(f"  Training set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
                logger.info(f"  Test set: {len(test_df)} records ({len(test_df)/len(df):.1%})")
            else:
                raise
        
        # Post-split quality validation
        logger.info("🔍 Performing post-split quality validation...")
        
        # Check target distribution in splits
        train_target_ratio = train_df[target_col].mean()
        test_target_ratio = test_df[target_col].mean()
        
        logger.info(f"Target distribution after split:")
        logger.info(f"  Training set churn rate: {train_target_ratio:.1%}")
        logger.info(f"  Test set churn rate: {test_target_ratio:.1%}")
        logger.info(f"  Original churn rate: {target_ratio:.1%}")
        
        # Check if distributions are similar (within 2 percentage points)
        distribution_diff = abs(train_target_ratio - test_target_ratio)
        if distribution_diff > 0.02:
            logger.warning(f"Target distribution differs significantly between splits: {distribution_diff:.1%}")
        else:
            logger.info("✅ Target distributions are well balanced between splits")
        
        # Check feature distribution consistency (for numeric features)
        distribution_checks = []
        for feature in numeric_features[:5]:  # Check first 5 numeric features
            train_mean = train_df[feature].mean()
            test_mean = test_df[feature].mean()
            
            # Calculate relative difference
            if train_mean != 0:
                relative_diff = abs(train_mean - test_mean) / abs(train_mean)
                distribution_checks.append((feature, relative_diff))
                
                if relative_diff > 0.1:  # More than 10% difference
                    logger.warning(f"Feature '{feature}' has different distributions: train={train_mean:.3f}, test={test_mean:.3f}")
        
        if distribution_checks:
            avg_diff = np.mean([diff for _, diff in distribution_checks])
            logger.info(f"Average feature distribution difference: {avg_diff:.1%}")
            
            if avg_diff < 0.05:  # Less than 5% average difference
                logger.info("✅ Feature distributions are consistent between splits")
        
        # Data leakage check
        logger.info("🔒 Checking for potential data leakage...")
        
        # Check for duplicate rows between train and test
        if 'customer_id' in train_df.columns:
            train_ids = set(train_df['customer_id'])
            test_ids = set(test_df['customer_id'])
            overlap = train_ids.intersection(test_ids)
            
            if overlap:
                logger.error(f"❌ Data leakage detected: {len(overlap)} customer IDs appear in both train and test sets")
                raise ValueError("Data leakage detected - same customers in both train and test sets")
            else:
                logger.info("✅ No customer ID overlap between train and test sets")
        
        # Final split summary
        logger.info("📊 Final split summary:")
        logger.info(f"  Original dataset: {len(df)} records, {len(df.columns)} features")
        logger.info(f"  Training set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
        logger.info(f"  Test set: {len(test_df)} records ({len(test_df)/len(df):.1%})")
        logger.info(f"  Features: {len(feature_columns)}")
        
        # Save split data based on execution mode
        if is_local_mode:
            # LOCAL MODE: Save to data/ folder
            logger.info("💾 Saving split data to local data/ folder...")
            
            data_dir = "data/04_train_test_split/"
            os.makedirs(data_dir, exist_ok=True)
            
            # Save training data
            train_filename = f"train_data_{train_feature_view}.csv"
            train_filepath = os.path.join(data_dir, train_filename)
            train_df.to_csv(train_filepath, index=False)
            
            # Save test data
            test_filename = f"test_data_{test_feature_view}.csv"
            test_filepath = os.path.join(data_dir, test_filename)
            test_df.to_csv(test_filepath, index=False)
            
            # Save split metadata
            metadata = {
                'execution_mode': 'local',
                'timestamp': datetime.now().isoformat(),
                'source_feature_view': source_feature_view,
                'source_version': source_version,
                'train_feature_view': train_feature_view,
                'test_feature_view': test_feature_view,
                'output_version': output_version,
                'original_records': len(df),
                'train_records': len(train_df),
                'test_records': len(test_df),
                'test_size': test_size,
                'random_state': random_state,
                'stratified': stratify,
                'original_churn_rate': target_ratio,
                'train_churn_rate': train_target_ratio,
                'test_churn_rate': test_target_ratio,
                'features_count': len(feature_columns),
                'train_file_path': train_filepath,
                'test_file_path': test_filepath
            }
            
            metadata_file = os.path.join(data_dir, f"split_metadata_{output_version}.json")
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Training data saved to: {train_filepath}")
            logger.info(f"✅ Test data saved to: {test_filepath}")
            logger.info(f"✅ Metadata saved to: {metadata_file}")
            
        else:
            # DEPLOYMENT MODE: Save to Snowflake Feature Store
            logger.info("☁️  Saving split data to Snowflake Feature Store...")
            
            # Create separate FeatureViews for train and test data
            from snowflake.ml.feature_store import FeatureView, Entity
            
            train_feature_view_obj = FeatureView(
                name=train_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=train_df,
                timestamp_col="created_date",
                desc=f"Training data with {len(train_df)} records"
            )
            feature_store.register_feature_view(train_feature_view_obj, version=output_version)

            test_feature_view_obj = FeatureView(
                name=test_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=test_df,
                timestamp_col="created_date",
                desc=f"Test data with {len(test_df)} records"
            )
            feature_store.register_feature_view(test_feature_view_obj, version=output_version)
            
            logger.info(f"✅ Feature views '{train_feature_view}:{output_version}' & '{test_feature_view}:{output_version}' created in Snowflake Feature Store")
            logger.info(f"📊 Registered {len(train_df)} training records and {len(test_df)} test records")
        
        logger.info(f"🎯 Train/test split completed successfully in {mode} mode")
        logger.info(f"📊 Split {len(df)} records into {len(train_df)} train + {len(test_df)} test")
        
    except Exception as e:
        logger.error(f"❌ Failed to split data: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing create_train_test_split node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "source_feature_view": "test_engineered_customer_features",
        "version": "v1",
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True
    }
    
    test_outputs = {
        "train_feature_view": "test_train_features",
        "test_feature_view": "test_test_features",
        "version": "v1"
    }
    
    # Run in local mode (feature_store=None)
    create_train_test_split(
        feature_store=None,  # Local mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    # Test deployment mode simulation
    print("\n=== Testing DEPLOYMENT mode simulation ===")
    
    # Create a mock feature store object
    class MockFeatureStore:
        def get_feature_view(self, name, version):
            print(f"Mock: Would get feature view {name}:{version}")
            return None
        
        def register_feature_view(self, feature_view, version):
            print(f"Mock: Would register {feature_view.name}:{version}")
    
    # Run in deployment mode (feature_store provided)
    create_train_test_split(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 