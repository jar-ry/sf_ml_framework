"""
Node 02: Preprocess Data

This node preprocesses the raw customer data by handling missing values,
outliers, and creating feature flags.
In local mode, reads/saves to data/ folder. In deployment mode, uses Snowflake Feature Store.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
from src.utils.data_catalog import get_catalog
from snowflake.ml.feature_store import FeatureView, Entity

logger = logging.getLogger(__name__)


def preprocess_data(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """
    Preprocess customer data by joining multiple datasets and applying transformations.
    
    Detects execution mode:
    - Local mode (feature_store=None): Reads from data/ folder, saves to data/ folder
    - Deployment mode (feature_store provided): Uses Snowflake Feature Store
    
    Args:
        feature_store: Snowflake Feature Store instance (None for local execution)
        model_registry: Snowflake Model Registry instance (not used in this node)
        inputs: Dictionary containing:
            - transactions_dataset: Name of the transactions dataset (default: "customer_transactions")
            - demographics_dataset: Name of the demographics dataset (default: "customer_demographics")
        outputs: Dictionary containing:
            - output_dataset: Name of the output dataset (default: "preprocessed_customer_features")
            - version: Version of the output dataset (default: "v1")
    
    Returns:
        None
    """
    
    # Detect execution mode
    is_local_mode = feature_store is None
    mode = "LOCAL" if is_local_mode else "DEPLOYMENT"
    logger.info(f"🔄 Running in {mode} mode")
    
    try:
        # Extract parameters and get catalog info
        catalog = get_catalog()
        transactions_dataset = inputs.get('transactions_dataset', 'customer_transactions')
        demographics_dataset = inputs.get('demographics_dataset', 'customer_demographics')
        
        # Use catalog's default versions if not specified
        transactions_version = inputs.get('transactions_version')
        if transactions_version is None:
            transactions_version = catalog.get_default_version(transactions_dataset)
        
        demographics_version = inputs.get('demographics_version')
        if demographics_version is None:
            demographics_version = catalog.get_default_version(demographics_dataset)
        
        output_dataset = outputs.get('output_dataset', 'preprocessed_customer_features')
        output_version = outputs.get('version')
        if output_version is None:
            output_version = catalog.get_default_version(output_dataset)
        
        # Get feature view names from catalog
        transactions_feature_view = catalog.get_snowflake_feature_view(transactions_dataset)
        demographics_feature_view = catalog.get_snowflake_feature_view(demographics_dataset)
        output_feature_view = catalog.get_snowflake_feature_view(output_dataset)
        
        logger.info(f"🔗 Joining multiple datasets:")
        logger.info(f"  Transactions: {transactions_dataset}:{transactions_version}")
        logger.info(f"  Demographics: {demographics_dataset}:{demographics_version}")
        logger.info(f"  Output: {output_dataset}:{output_version}")
        
        # Load both datasets based on execution mode
        if is_local_mode:
            # LOCAL MODE: Read from data/ folder using catalog
            logger.info("📁 Loading datasets from local data/ folder...")
            
            # Load transactions dataset
            transactions_filepath = catalog.find_local_file(transactions_dataset, version=transactions_version)
            if transactions_filepath and os.path.exists(transactions_filepath):
                transactions_df = pd.read_csv(transactions_filepath)
                logger.info(f"✅ Loaded transactions data from: {transactions_filepath}")
            else:
                raise FileNotFoundError(f"Could not find transactions data for {transactions_dataset}:{transactions_version}")
            
            # Load demographics dataset
            demographics_filepath = catalog.find_local_file(demographics_dataset, version=demographics_version)
            if demographics_filepath and os.path.exists(demographics_filepath):
                demographics_df = pd.read_csv(demographics_filepath)
                logger.info(f"✅ Loaded demographics data from: {demographics_filepath}")
            else:
                raise FileNotFoundError(f"Could not find demographics data for {demographics_dataset}:{demographics_version}")
                
        else:
            # DEPLOYMENT MODE: Read from Snowflake Feature Store
            logger.info("☁️  Loading datasets from Snowflake Feature Store...")
            
            # Load transactions dataset
            transactions_feature_view_obj = feature_store.get_feature_view(transactions_feature_view, transactions_version)
            transactions_df = transactions_feature_view_obj.to_pandas()
            logger.info(f"✅ Loaded transactions data from Snowflake: {transactions_feature_view}:{transactions_version}")
            
            # Load demographics dataset
            demographics_feature_view_obj = feature_store.get_feature_view(demographics_feature_view, demographics_version)
            demographics_df = demographics_feature_view_obj.to_pandas()
            logger.info(f"✅ Loaded demographics data from Snowflake: {demographics_feature_view}:{demographics_version}")
        
        logger.info(f"📊 Dataset sizes before join:")
        logger.info(f"  Transactions: {len(transactions_df)} records, {len(transactions_df.columns)} columns")
        logger.info(f"  Demographics: {len(demographics_df)} records, {len(demographics_df.columns)} columns")
        
        # Join the datasets on customer_id
        logger.info("🔗 Joining datasets on customer_id...")
        
        # Perform inner join to ensure we only keep customers present in both datasets
        df = pd.merge(
            transactions_df,
            demographics_df,
            on='customer_id',
            how='inner',
            suffixes=('_trans', '_demo')
        )
        
        logger.info(f"✅ Join completed: {len(df)} records with {len(df.columns)} columns")
        
        # Validate join quality
        original_customers = set(transactions_df['customer_id'])
        joined_customers = set(df['customer_id'])
        join_rate = len(joined_customers) / len(original_customers)
        
        logger.info(f"📊 Join quality:")
        logger.info(f"  Original customers: {len(original_customers)}")
        logger.info(f"  Joined customers: {len(joined_customers)}")
        logger.info(f"  Join rate: {join_rate:.1%}")
        
        if join_rate < 0.95:  # Less than 95% join rate
            logger.warning(f"Low join rate ({join_rate:.1%}) - some customers missing from demographics data")
        
        # Data quality checks before preprocessing
        logger.info("🔍 Performing initial data quality checks...")
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values across columns")
            for col, missing_count in missing_summary[missing_summary > 0].items():
                logger.warning(f"  {col}: {missing_count} missing ({missing_count/len(df):.1%})")
        else:
            logger.info("✅ No missing values found")
        
        # Check data types
        logger.info("Data types after join:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Preprocessing steps
        logger.info("🧹 Starting data preprocessing...")
        
        # 1. Handle missing values
        logger.info("Step 1: Handling missing values...")
        
        # Numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'customer_id' in numerical_cols:
            numerical_cols.remove('customer_id')  # Don't impute customer_id
        
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} missing values with median: {median_val}")
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col == 'customer_id':  # Skip customer_id
                continue
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"  Filled {col} missing values with mode: {mode_val}")
        
        # 2. Handle outliers
        logger.info("Step 2: Handling outliers...")
        
        outlier_columns = ['monthly_spend', 'age', 'sessions_per_month']
        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"  Capped {outlier_count} outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                else:
                    logger.info(f"  No outliers found in {col}")
        
        # 3. Data validation
        logger.info("Step 3: Data validation...")
        
        # Ensure age is reasonable
        if 'age' in df.columns:
            invalid_ages = df[(df['age'] < 18) | (df['age'] > 100)]
            if len(invalid_ages) > 0:
                df.loc[df['age'] < 18, 'age'] = 18
                df.loc[df['age'] > 100, 'age'] = 100
                logger.info(f"  Corrected {len(invalid_ages)} invalid ages")
        
        # Ensure monthly spend is non-negative
        if 'monthly_spend' in df.columns:
            negative_spend = df[df['monthly_spend'] < 0]
            if len(negative_spend) > 0:
                df.loc[df['monthly_spend'] < 0, 'monthly_spend'] = 0
                logger.info(f"  Corrected {len(negative_spend)} negative spending values")
        
        # 4. Feature engineering from joined data
        logger.info("Step 4: Creating enhanced features from joined data...")
        
        # High value customer flag (top 25% by spending)
        if 'monthly_spend' in df.columns:
            spend_threshold = df['monthly_spend'].quantile(0.75)
            df['is_high_value'] = (df['monthly_spend'] >= spend_threshold).astype(int)
            high_value_count = df['is_high_value'].sum()
            logger.info(f"  Created is_high_value flag: {high_value_count} customers ({high_value_count/len(df):.1%})")
        
        # High support ticket flag (more than 2 tickets)
        if 'support_tickets_last_month' in df.columns:
            df['is_high_support'] = (df['support_tickets_last_month'] > 2).astype(int)
            high_support_count = df['is_high_support'].sum()
            logger.info(f"  Created is_high_support flag: {high_support_count} customers ({high_support_count/len(df):.1%})")
        
        # Low engagement flag (bottom 25% by sessions)
        if 'sessions_per_month' in df.columns:
            session_threshold = df['sessions_per_month'].quantile(0.25)
            df['is_low_engagement'] = (df['sessions_per_month'] <= session_threshold).astype(int)
            low_engagement_count = df['is_low_engagement'].sum()
            logger.info(f"  Created is_low_engagement flag: {low_engagement_count} customers ({low_engagement_count/len(df):.1%})")
        
        # Regional features (if region is available)
        if 'region' in df.columns:
            # Create region dummy variables
            region_dummies = pd.get_dummies(df['region'], prefix='region')
            df = pd.concat([df, region_dummies], axis=1)
            logger.info(f"  Created {len(region_dummies.columns)} region indicator features")
        
        # Account age features (if account_created_date is available)
        if 'account_created_date' in df.columns:
            from datetime import datetime
            try:
                df['account_created_date'] = pd.to_datetime(df['account_created_date'])
                current_date = datetime.now()
                df['account_age_days'] = (current_date - df['account_created_date']).dt.days
                df['account_age_years'] = df['account_age_days'] / 365.25
                
                # Create account age buckets
                df['account_age_bucket'] = pd.cut(
                    df['account_age_years'],
                    bins=[0, 1, 3, 5, float('inf')],
                    labels=['New', 'Young', 'Mature', 'Veteran']
                )
                
                logger.info(f"  Created account age features")
                logger.info(f"  Account age distribution: {df['account_age_bucket'].value_counts().to_dict()}")
            except Exception as e:
                logger.warning(f"  Failed to create account age features: {str(e)}")
        
        # 5. Final data quality summary
        logger.info("📊 Final preprocessed data summary:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Total features: {len(df.columns)}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean()
            logger.info(f"  Churn rate: {churn_rate:.1%}")
        
        # List all final columns
        logger.info(f"  Final columns: {list(df.columns)}")
        
        # Save processed data based on execution mode
        if is_local_mode:
            # LOCAL MODE: Save to data/ folder using catalog
            logger.info("💾 Saving preprocessed data to local data/ folder...")
            
            # Get paths from catalog
            filepath = catalog.get_local_path(output_dataset, create_dir=True)
            metadata_file = catalog.get_local_metadata_path(output_dataset, version=output_version, create_dir=True)
            
            # Save processed data
            df.to_csv(filepath, index=False)
            
            # Save preprocessing metadata
            metadata = {
                'execution_mode': 'local',
                'timestamp': datetime.now().isoformat(),
                'transactions_dataset': transactions_dataset,
                'demographics_dataset': demographics_dataset,
                'transactions_version': transactions_version,
                'demographics_version': demographics_version,
                'output_dataset': output_dataset,
                'output_feature_view': output_feature_view,
                'output_version': output_version,
                'input_transactions_records': len(transactions_df),
                'input_demographics_records': len(demographics_df),
                'output_records': len(df),
                'join_rate': join_rate,
                'features_added': [
                    'is_high_value', 'is_high_support', 'is_low_engagement',
                    'region_*', 'account_age_days', 'account_age_years', 'account_age_bucket'
                ],
                'preprocessing_steps': [
                    'dataset_joining',
                    'missing_value_imputation',
                    'outlier_capping', 
                    'data_validation',
                    'enhanced_feature_creation'
                ],
                'file_path': filepath
            }
            
            # metadata_file already defined above from catalog
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Preprocessed data saved to: {filepath}")
            logger.info(f"✅ Metadata saved to: {metadata_file}")
            
        else:
            # DEPLOYMENT MODE: Save to Snowflake Feature Store
            logger.info("☁️  Saving preprocessed data to Snowflake Feature Store...")
            
            # Create FeatureView for processed data
            from snowflake.ml.feature_store import FeatureView, Entity
            
            processed_feature_view = FeatureView(
                name=output_feature_view,
                entities=[Entity(name="customer_id", join_keys=["customer_id"])],
                feature_df=df,
                timestamp_col="account_created_date" if "account_created_date" in df.columns else None,
                desc=f"Preprocessed customer data joined from {transactions_dataset} and {demographics_dataset} with {len(df)} records"
            )
            
            feature_store.register_feature_view(processed_feature_view, version=output_version)
            
            logger.info(f"✅ Feature view '{output_feature_view}:{output_version}' created in Snowflake Feature Store")
            logger.info(f"📊 Registered {len(df)} preprocessed records with {len(df.columns)} features")
        
        logger.info(f"🎯 Data preprocessing completed successfully in {mode} mode")
        logger.info(f"📊 Joined and processed {len(df)} records with {len(df.columns)} features")
        logger.info(f"🔗 Successfully demonstrated multi-dataset joining with versions {transactions_version} & {demographics_version}")
        
    except Exception as e:
        logger.error(f"❌ Failed to preprocess data: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the node function locally
    print("Testing preprocess_data node...")
    
    # Test local mode
    print("\n=== Testing LOCAL mode ===")
    test_inputs = {
        "transactions_dataset": "customer_transactions",
        "demographics_dataset": "customer_demographics",
        "transactions_version": "v1",
        "demographics_version": "v2"
    }
    
    test_outputs = {
        "output_dataset": "test_preprocessed_customer_features",
        "version": "v1"
    }
    
    # Run in local mode (feature_store=None)
    preprocess_data(
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
    preprocess_data(
        feature_store=MockFeatureStore(),  # Deployment mode
        model_registry=None,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print("✅ Node test completed successfully!") 