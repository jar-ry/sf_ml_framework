#!/usr/bin/env python3
"""
End-to-end pipeline test script

This script tests the complete MLOps pipeline locally to ensure
all nodes work correctly together.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_test_environment():
    """Set up the test environment."""
    logger.info("Setting up test environment...")
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Clean up any existing test files
    test_files = [
        "generated_data_raw_customer_data.csv",
        "preprocessed_data_preprocessed_customer_features.csv", 
        "engineered_data_engineered_customer_features.csv",
        "train_data_train_customer_features.csv",
        "test_data_test_customer_features.csv",
        "split_metadata_v1.json",
        "model_customer_churn_xgboost_v1.pkl",
        "features_customer_churn_xgboost_v1.json",
        "metadata_customer_churn_xgboost_v1.json",
        "feature_importance_customer_churn_xgboost_v1.csv",
        "evaluation_results_customer_churn_xgboost_v1.json",
        "evaluation_summary_customer_churn_xgboost_v1.csv",
        "shap_summary_customer_churn_xgboost_v1.png",
        "shap_importance_customer_churn_xgboost_v1.png",
        "roc_curve_customer_churn_xgboost_v1.png",
        "pr_curve_customer_churn_xgboost_v1.png",
        "confusion_matrix_customer_churn_xgboost_v1.png"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Removed existing test file: {file}")


def test_pipeline_validation():
    """Test pipeline configuration validation."""
    logger.info("Testing pipeline validation...")
    
    try:
        from pipeline import create_pipeline, validate_pipeline, get_execution_order
        
        # Test pipeline creation
        pipeline = create_pipeline()
        assert len(pipeline) == 6, f"Expected 6 nodes, got {len(pipeline)}"
        logger.info(f"✅ Pipeline created with {len(pipeline)} nodes")
        
        # Test validation
        errors = validate_pipeline()
        if errors:
            logger.warning(f"Pipeline validation issues: {errors}")
        else:
            logger.info("✅ Pipeline validation passed")
        
        # Test execution order
        order = get_execution_order()
        expected_order = [
            "generate_sample_data",
            "preprocess_data", 
            "engineer_features",
            "create_train_test_split",
            "train_model",
            "evaluate_model"
        ]
        assert order == expected_order, f"Expected {expected_order}, got {order}"
        logger.info(f"✅ Execution order correct: {' → '.join(order)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_node_execution(node_name, node_function, inputs, outputs):
    """Test execution of a single node."""
    logger.info(f"Testing node: {node_name}")
    
    try:
        node_function(
            feature_store=None,
            model_registry=None,
            inputs=inputs,
            outputs=outputs
        )
        logger.info(f"✅ Node {node_name} executed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Node {node_name} failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_complete_pipeline():
    """Test the complete pipeline end-to-end."""
    logger.info("Testing complete pipeline execution...")
    
    try:
        # Import node functions
        from nodes.node_01_generate_data import generate_sample_data
        from nodes.node_02_preprocess_data import preprocess_data
        from nodes.node_03_feature_engineering import engineer_features
        from nodes.node_04_train_test_split import create_train_test_split
        from nodes.node_05_train_model import train_model
        from nodes.node_06_evaluate_model import evaluate_model
        
        success_count = 0
        total_nodes = 6
        
        # Node 1: Generate Sample Data
        if test_node_execution(
            "generate_sample_data",
            generate_sample_data,
            inputs={
                "customer_count": 100000,
                "churn_rate": 0.12,
                "random_seed": 42
            },
            outputs={
                "feature_view_name": "raw_customer_data",
                "version": "v1"
            }
        ):
            success_count += 1
        
        # Node 2: Preprocess Data
        if test_node_execution(
            "preprocess_data", 
            preprocess_data,
            inputs={
                "source_feature_view": "raw_customer_data",
                "version": "v1"
            },
            outputs={
                "feature_view_name": "preprocessed_customer_features",
                "version": "v1"
            }
        ):
            success_count += 1
        
        # Node 3: Engineer Features
        if test_node_execution(
            "engineer_features",
            engineer_features,
            inputs={
                "source_feature_view": "preprocessed_customer_features",
                "version": "v1"
            },
            outputs={
                "feature_view_name": "engineered_customer_features", 
                "version": "v1"
            }
        ):
            success_count += 1
        
        # Node 4: Train/Test Split
        if test_node_execution(
            "create_train_test_split",
            create_train_test_split,
            inputs={
                "source_feature_view": "engineered_customer_features",
                "version": "v1",
                "test_size": 0.2,
                "random_seed": 42,
                "stratify_columns": ["churned"]
            },
            outputs={
                "train_feature_view": "train_customer_features",
                "test_feature_view": "test_customer_features",
                "version": "v1"
            }
        ):
            success_count += 1
        
        # Node 5: Train Model
        if test_node_execution(
            "train_model",
            train_model,
            inputs={
                "train_feature_view": "train_customer_features",
                "version": "v1",
                "target_column": "churned",
                "model_params": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "n_estimators": 50,
                    "random_state": 42
                },
                "cv_folds": 3  # Reduced for testing
            },
            outputs={
                "model_name": "customer_churn_xgboost",
                "model_version": "v1"
            }
        ):
            success_count += 1
        
        # Node 6: Evaluate Model
        if test_node_execution(
            "evaluate_model",
            evaluate_model,
            inputs={
                "model_name": "customer_churn_xgboost",
                "model_version": "v1",
                "test_feature_view": "test_customer_features",
                "version": "v1",
                "target_column": "churned"
            },
            outputs={
                "evaluation_table": "MODEL_EVALUATION_RESULTS",
                "shap_plots_stage": "SHAP_PLOTS"
            }
        ):
            success_count += 1
        
        logger.info(f"\n📊 Pipeline Test Results:")
        logger.info(f"Total nodes: {total_nodes}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {total_nodes - success_count}")
        logger.info(f"Success rate: {success_count/total_nodes:.1%}")
        
        return success_count == total_nodes
        
    except ImportError as e:
        logger.error(f"❌ Could not import node functions: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def verify_output_files():
    """Verify that expected output files were created."""
    logger.info("Verifying output files...")
    
    expected_files = [
        "generated_data_raw_customer_data.csv",
        "preprocessed_data_preprocessed_customer_features.csv",
        "engineered_data_engineered_customer_features.csv", 
        "train_data_train_customer_features.csv",
        "test_data_test_customer_features.csv",
        "model_customer_churn_xgboost_v1.pkl",
        "evaluation_results_customer_churn_xgboost_v1.json"
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            size = os.path.getsize(file)
            logger.info(f"✅ {file} ({size} bytes)")
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    logger.info("✅ All expected output files created")
    return True


def run_tests():
    """Run all tests."""
    logger.info("🚀 Starting MLOps Pipeline Tests")
    logger.info("=" * 60)
    
    # Setup
    setup_test_environment()
    
    # Test pipeline configuration
    config_ok = test_pipeline_validation()
    
    # Test complete pipeline execution
    pipeline_ok = test_complete_pipeline()
    
    # Verify outputs
    files_ok = verify_output_files()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 Test Summary:")
    logger.info(f"Pipeline Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    logger.info(f"Pipeline Execution: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    logger.info(f"Output Files: {'✅ PASS' if files_ok else '❌ FAIL'}")
    
    overall_success = config_ok and pipeline_ok and files_ok
    logger.info(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the MLOps pipeline')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test files after running')
    args = parser.parse_args()
    
    success = run_tests()
    
    if args.cleanup:
        logger.info("Cleaning up test files...")
        setup_test_environment()  # Reuse cleanup logic
        logger.info("✅ Cleanup completed")
    
    sys.exit(0 if success else 1) 