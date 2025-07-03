"""
Node 03: Feature Engineering

This node performs advanced feature engineering on preprocessed customer data.
In local mode, reads/saves to data/ folder. In deployment mode, uses Snowflake Feature Store.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.data_catalog import get_catalog

logger = logging.getLogger(__name__)


def engineer_features(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Perform advanced feature engineering on customer data.
    
    Detects execution mode:
    - Local mode (feature_store=None): Reads from data/ folder, saves to data/ folder
    - Deployment mode (feature_store provided): Uses Snowflake Feature Store
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (not used in this node)
        inputs: Dictionary containing:
            - source_feature_view: Name of the preprocessed feature view
            - version: Version of the source feature view
        outputs: Dictionary containing:
            - feature_view_name: Name of the engineered feature view
            - version: Version of the output feature view
    
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
        
        source_feature_view = inputs.get('source_feature_view', 'preprocessed_customer_features')
        output_feature_view = outputs.get('feature_view_name', 'engineered_customer_features')
        
        # Get versions from catalog instead of hardcoded defaults
        source_version = catalog.get_default_version(source_feature_view)
        output_version = catalog.get_default_version(output_feature_view)
        
        logger.info(f"Engineering features: {source_feature_view}:{source_version} → {output_feature_view}:{output_version}")
        
        # Load data based on execution mode
        if is_local_mode:
            # LOCAL MODE: Read from data/ folder
            logger.info("📁 Loading data from local data/ folder...")
            
            # Try multiple possible file locations
            possible_files = [
                f"data/02_preprocess_data/preprocessed_data_{source_feature_view}.csv",
                f"data/preprocessed_data_{source_feature_view}.csv",
                f"preprocessed_data_{source_feature_view}.csv",
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
        
        # Simple feature engineering for demo (3 new features only)
        logger.info("🔧 Starting simple feature engineering...")
        original_features = len(df.columns)
        
        # Feature 1: Spend per session (how much they spend per interaction)
        if all(col in df.columns for col in ['monthly_spend', 'sessions_per_month']):
            df['spend_per_session'] = np.where(
                df['sessions_per_month'] > 0,
                df['monthly_spend'] / df['sessions_per_month'],
                0
            )
            logger.info("  ✅ Created spend_per_session")
        
        # Feature 2: Support risk score (normalized support tickets)
        if 'support_tickets_last_month' in df.columns:
            max_tickets = df['support_tickets_last_month'].max()
            if max_tickets > 0:
                df['support_risk_score'] = df['support_tickets_last_month'] / max_tickets
            else:
                df['support_risk_score'] = 0
            logger.info("  ✅ Created support_risk_score")
        
        # Feature 3: Customer value tier (combination of spend and engagement)
        if all(col in df.columns for col in ['monthly_spend', 'sessions_per_month']):
            # Normalize to 0-1 scale
            spend_min, spend_max = df['monthly_spend'].min(), df['monthly_spend'].max()
            sessions_min, sessions_max = df['sessions_per_month'].min(), df['sessions_per_month'].max()
            
            if spend_max > spend_min:
                spend_normalized = (df['monthly_spend'] - spend_min) / (spend_max - spend_min)
            else:
                spend_normalized = 0
                
            if sessions_max > sessions_min:
                sessions_normalized = (df['sessions_per_month'] - sessions_min) / (sessions_max - sessions_min)
            else:
                sessions_normalized = 0
            
            # Create composite score (equal weights)
            df['customer_value_tier'] = (spend_normalized + sessions_normalized) / 2
            logger.info("  ✅ Created customer_value_tier")
        
        # Handle categorical columns for XGBoost compatibility
        logger.info("🔧 Handling categorical columns...")
        
        # Handle customer_id (label encode)
        if 'customer_id' in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['customer_id_encoded'] = le.fit_transform(df['customer_id'].astype(str))
            df = df.drop(columns=['customer_id'])
            logger.info("  ✅ Encoded customer_id")
        
        # Remove any infinite or NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        initial_nans = df[numeric_columns].isna().sum().sum()
        if initial_nans > 0:
            df[numeric_columns] = df[numeric_columns].fillna(0)
            logger.info(f"  ✅ Filled {initial_nans} NaN values")
        
        # Ensure no object columns remain
        remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if remaining_object_cols:
            logger.warning(f"  Still have object columns: {remaining_object_cols}")
            df = df.select_dtypes(exclude=['object'])
            logger.info(f"  ✅ Dropped remaining object columns for XGBoost compatibility")
        
        # Final summary
        new_features = len(df.columns) - original_features
        logger.info("📊 Feature engineering summary:")
        logger.info(f"  Original features: {original_features}")
        logger.info(f"  New features created: {new_features}")
        logger.info(f"  Total features: {len(df.columns)}")
        
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean()
            logger.info(f"  Churn rate: {churn_rate:.1%}")
        
        # Save engineered features based on execution mode
        if is_local_mode:
            # LOCAL MODE: Save to data/ folder
            logger.info("💾 Saving engineered features to local data/ folder...")
            
            data_dir = "data/03_feature_engineering/"
            os.makedirs(data_dir, exist_ok=True)
            
            # Save engineered data
            filename = f"engineered_features_{output_feature_view}.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False)
            
            # Save feature engineering metadata
            metadata = {
                'execution_mode': 'local',
                'timestamp': datetime.now().isoformat(),
                'source_feature_view': source_feature_view,
                'source_version': source_version,
                'output_feature_view': output_feature_view,
                'output_version': output_version,
                'input_records': len(df),
                'input_features': original_features,
                'output_features': len(df.columns),
                'new_features_created': new_features,
                'feature_engineering_steps': [
                    'spend_per_session',
                    'support_risk_score', 
                    'customer_value_tier',
                    'categorical_encoding'
                ],
                'file_path': filepath
            }
            
            metadata_file = os.path.join(data_dir, f"feature_engineering_metadata_{output_feature_view}_{output_version}.json")
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Engineered features saved to: {filepath}")
            logger.info(f"✅ Metadata saved to: {metadata_file}")
            
        else:
            # DEPLOYMENT MODE: Save to Snowflake Feature Store
            logger.info("☁️  Saving engineered features to Snowflake Feature Store...")
            
            # Create FeatureView for engineered features
            from snowflake.ml.feature_store import FeatureView, Entity
            
            feature_view = FeatureView(
                name=output_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=df,
                timestamp_col="created_date",
                desc=f"Engineered customer features with {new_features} new features"
            )
            
            # Register the feature view
            feature_store.register_feature_view(feature_view, version=output_version)
            
            logger.info(f"✅ Feature view '{output_feature_view}:{output_version}' created in Snowflake Feature Store")
            logger.info(f"📊 Registered {len(df)} records with {new_features} new features")
        
        logger.info(f"🎯 Feature engineering completed successfully in {mode} mode")
        logger.info(f"📊 Created {new_features} new features from {original_features} original features")
        
    except Exception as e:
        logger.error(f"❌ Failed to engineer features: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing engineer_features node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "source_feature_view": "test_preprocessed_customer_features",
        "version": "v1"
    }
    
    test_outputs = {
        "feature_view_name": "test_engineered_customer_features",
        "version": "v1"
    }
    
    # Run in local mode (feature_store=None)
    engineer_features(
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
    engineer_features(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 