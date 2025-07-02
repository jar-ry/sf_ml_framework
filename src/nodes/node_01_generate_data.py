"""
Node 01: Generate Sample Data

This node generates synthetic customer data for the churn prediction pipeline.
In local mode, saves to data/ folder. In deployment mode, saves to Snowflake Feature Store.
"""

import logging
import os
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from src.data_catalog import get_catalog

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add data directory to path for sample_customer_data import
data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
sys.path.append(data_dir_path)

from sample_customer_data import generate_customer_data

logger = logging.getLogger(__name__)


def generate_sample_data(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Generate synthetic customer data split into two datasets for joining demonstration.
    
    Detects execution mode:
    - Local mode (feature_store=None): Saves to data/ folder as CSV
    - Deployment mode (feature_store provided): Saves to Snowflake Feature Store
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (not used in this node)
        inputs: Dictionary containing:
            - customer_count: Number of customers to generate (default: 100000)
            - churn_rate: Percentage of customers who churned (default: 0.12)
            - random_seed: Random seed for reproducibility (default: 42)
        outputs: Dictionary containing:
            - transactions_dataset: Name of the transactions dataset (default: "customer_transactions")
            - demographics_dataset: Name of the demographics dataset (default: "customer_demographics")
    
    Returns:
        None
    """
    
    # Detect execution mode
    is_local_mode = feature_store is None
    mode = "LOCAL" if is_local_mode else "DEPLOYMENT"
    logger.info(f"🔄 Running in {mode} mode")
    
    try:
        # Extract parameters with defaults
        customer_count = inputs.get('customer_count', 100000)
        churn_rate = inputs.get('churn_rate', 0.12)
        random_seed = inputs.get('random_seed', 42)
        
        # Get data catalog info for both datasets
        catalog = get_catalog()
        transactions_dataset = outputs.get('transactions_dataset', 'customer_transactions')
        demographics_dataset = outputs.get('demographics_dataset', 'customer_demographics')
        
        # Get versions from catalog instead of outputs
        transactions_version = catalog.get_default_version(transactions_dataset)
        demographics_version = catalog.get_default_version(demographics_dataset)
        
        transactions_feature_view = catalog.get_snowflake_feature_view(transactions_dataset)
        demographics_feature_view = catalog.get_snowflake_feature_view(demographics_dataset)
        
        logger.info(f"Generating {customer_count} customers with {churn_rate:.1%} churn rate")
        logger.info(f"Creating two datasets:")
        logger.info(f"  Transactions: {transactions_dataset}:{transactions_version}")
        logger.info(f"  Demographics: {demographics_dataset}:{demographics_version}")
        
        # Generate the complete customer data using our existing function
        full_df = generate_customer_data(
            num_customers=customer_count,
            churn_rate=churn_rate,
            random_seed=random_seed
        )
        
        logger.info(f"Generated {len(full_df)} customer records with {len(full_df.columns)} features")
        
        # Split the data into two datasets
        # Transactions dataset (v1): customer_id, behavioral data, target
        transactions_df = full_df[[
            'customer_id', 
            'monthly_spend', 
            'sessions_per_month', 
            'support_tickets_last_month',
            'churned'
        ]].copy()
        
        # Demographics dataset (v2): customer_id, demographic data
        demographics_df = full_df[[
            'customer_id',
            'age'
        ]].copy()
        
        # Add some additional demographic fields to make it more realistic
        import numpy as np
        np.random.seed(random_seed)
        
        # Add region (categorical)
        regions = ['North', 'South', 'East', 'West', 'Central']
        demographics_df['region'] = np.random.choice(regions, size=len(demographics_df))
        
        # Add account creation date (simulate account age)
        from datetime import datetime, timedelta
        base_date = datetime(2020, 1, 1)
        max_days = (datetime.now() - base_date).days
        random_days = np.random.randint(0, max_days, size=len(demographics_df))
        demographics_df['account_created_date'] = [
            (base_date + timedelta(days=int(days))).strftime('%Y-%m-%d') 
            for days in random_days
        ]
        
        logger.info(f"📊 Dataset split:")
        logger.info(f"  Transactions: {len(transactions_df)} records, {len(transactions_df.columns)} columns")
        logger.info(f"  Demographics: {len(demographics_df)} records, {len(demographics_df.columns)} columns")
        
        # Log data quality summary
        churn_count = transactions_df['churned'].sum()
        actual_churn_rate = churn_count / len(transactions_df)
        logger.info(f"Actual churn rate: {actual_churn_rate:.1%} ({churn_count}/{len(transactions_df)})")
        
        if is_local_mode:
            # LOCAL MODE: Save both datasets to data/ folder using catalog
            logger.info("💾 Saving datasets to local data/ folder...")
            
            # Save transactions dataset (v1)
            transactions_filepath = catalog.get_local_path(transactions_dataset, create_dir=True)
            transactions_metadata_file = catalog.get_local_metadata_path(
                transactions_dataset, version=transactions_version, create_dir=True
            )
            
            transactions_df.to_csv(transactions_filepath, index=False)
            
            # Save transactions metadata
            transactions_metadata = {
                'execution_mode': 'local',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': transactions_dataset,
                'feature_view_name': transactions_feature_view,
                'version': transactions_version,
                'customer_count': customer_count,
                'churn_rate': churn_rate,
                'random_seed': random_seed,
                'actual_churn_rate': actual_churn_rate,
                'file_path': transactions_filepath,
                'record_count': len(transactions_df),
                'feature_count': len(transactions_df.columns),
                'columns': list(transactions_df.columns),
                'dataset_type': 'transactions'
            }
            
            import json
            with open(transactions_metadata_file, 'w') as f:
                json.dump(transactions_metadata, f, indent=2)
            
            # Save demographics dataset (v2)
            demographics_filepath = catalog.get_local_path(demographics_dataset, create_dir=True)
            demographics_metadata_file = catalog.get_local_metadata_path(
                demographics_dataset, version=demographics_version, create_dir=True
            )
            
            demographics_df.to_csv(demographics_filepath, index=False)
            
            # Save demographics metadata
            demographics_metadata = {
                'execution_mode': 'local',
                'timestamp': datetime.now().isoformat(),
                'dataset_name': demographics_dataset,
                'feature_view_name': demographics_feature_view,
                'version': demographics_version,
                'customer_count': customer_count,
                'random_seed': random_seed,
                'file_path': demographics_filepath,
                'record_count': len(demographics_df),
                'feature_count': len(demographics_df.columns),
                'columns': list(demographics_df.columns),
                'dataset_type': 'demographics'
            }
            
            with open(demographics_metadata_file, 'w') as f:
                json.dump(demographics_metadata, f, indent=2)
            
            logger.info(f"✅ Transactions data saved to: {transactions_filepath}")
            logger.info(f"✅ Transactions metadata saved to: {transactions_metadata_file}")
            logger.info(f"✅ Demographics data saved to: {demographics_filepath}")
            logger.info(f"✅ Demographics metadata saved to: {demographics_metadata_file}")
            
        else:
            # DEPLOYMENT MODE: Save both datasets to Snowflake Feature Store
            logger.info("☁️  Saving datasets to Snowflake Feature Store...")
            
            # Create FeatureView for transactions dataset
            from snowflake.ml.feature_store import FeatureView, Entity
            
            transactions_feature_view_obj = FeatureView(
                name=transactions_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=transactions_df,
                desc=f"Customer transaction data with {len(transactions_df)} customers and {actual_churn_rate:.1%} churn rate"
            )
            
            feature_store.register_feature_view(transactions_feature_view_obj, version=transactions_version)
            
            # Create FeatureView for demographics dataset  
            demographics_feature_view_obj = FeatureView(
                name=demographics_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=demographics_df,
                desc=f"Customer demographic data with {len(demographics_df)} customers"
            )
            
            feature_store.register_feature_view(demographics_feature_view_obj, version=demographics_version)
            
            logger.info(f"✅ Feature view '{transactions_feature_view}:{transactions_version}' created in Snowflake Feature Store")
            logger.info(f"✅ Feature view '{demographics_feature_view}:{demographics_version}' created in Snowflake Feature Store")
            logger.info(f"📊 Registered {len(transactions_df)} transaction records and {len(demographics_df)} demographic records")
        
        logger.info(f"🎯 Data generation completed successfully in {mode} mode")
        logger.info(f"📊 Created two datasets with {len(full_df)} customers total")
        logger.info(f"📈 Churn rate: {actual_churn_rate:.1%}")
        logger.info(f"🔗 Datasets can be joined on 'customer_id'")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate sample data: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing generate_sample_data node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "customer_count": 50,
        "churn_rate": 0.15,
        "random_seed": 42
    }
    
    test_outputs = {
        "transactions_dataset": "customer_transactions",
        "demographics_dataset": "customer_demographics",
        "transactions_version": "v1",
        "demographics_version": "v2"
    }
    
    # Run in local mode (feature_store=None)
    generate_sample_data(
        feature_store=None,  # Local mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    # Test deployment mode simulation
    print("\n=== Testing DEPLOYMENT mode simulation ===")
    
    # Create a mock feature store object
    class MockFeatureStore:
        def register_feature_view(self, feature_view, version):
            print(f"Mock: Would register {feature_view.name}:{version}")
    
    # Run in deployment mode (feature_store provided)
    generate_sample_data(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 