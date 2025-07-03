"""
Pipeline definition and node registry for MLOps framework.

This module contains the central pipeline manifest that defines all nodes,
their functions, inputs, outputs, and execution metadata.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

# Import node functions with error handling
try:
    from src.nodes.node_01_generate_data import generate_sample_data
    from src.nodes.node_02_preprocess_data import preprocess_data
    from src.nodes.node_03_feature_engineering import engineer_features
    from src.nodes.node_04_train_test_split import create_train_test_split
    from src.nodes.node_05_train_model import train_model
    from src.nodes.node_06_evaluate_model import evaluate_model
    NODE_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some node modules not available: {e}")
    NODE_IMPORTS_AVAILABLE = False
    # Define placeholder functions
    def generate_sample_data(*args, **kwargs): pass
    def preprocess_data(*args, **kwargs): pass
    def engineer_features(*args, **kwargs): pass
    def create_train_test_split(*args, **kwargs): pass
    def train_model(*args, **kwargs): pass
    def evaluate_model(*args, **kwargs): pass


@dataclass
class NodeConfig:
    """Configuration for a pipeline node."""
    name: str
    description: str
    function: Callable
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    dependencies: List[str]
    compute_pool_size: str = "SMALL"
    memory_request: str = "2Gi"
    cpu_request: str = "1"
    timeout_minutes: int = 60


def create_pipeline() -> List[Dict[str, Any]]:
    """
    Create the complete customer churn prediction pipeline.
    
    Returns:
        List of node configurations defining the pipeline
    """
    
    # Define all pipeline nodes
    nodes = [
        NodeConfig(
            name="generate_sample_data",
            description="Generate synthetic customer data split into transactions and demographics",
            function=generate_sample_data,
            inputs={
                "customer_count": 100000,
                "churn_rate": 0.12,
                "random_seed": 42
            },
            outputs={
                "transactions_dataset": "customer_transactions",
                "demographics_dataset": "customer_demographics" 
            },
            dependencies=[],
            compute_pool_size="SMALL",
            memory_request="1Gi",
            cpu_request="1",
            timeout_minutes=30
        ),
        
        NodeConfig(
            name="preprocess_data",
            description="Join and preprocess customer transactions and demographics data",
            function=preprocess_data,
            inputs={
                "transactions_dataset": "customer_transactions",
                "demographics_dataset": "customer_demographics"
            },
            outputs={
                "output_dataset": "preprocessed_customer_features"
            },
            dependencies=["generate_sample_data"],
            compute_pool_size="SMALL",
            memory_request="2Gi",
            cpu_request="1",
            timeout_minutes=45
        ),
        
        NodeConfig(
            name="engineer_features",
            description="Create advanced features for model training",
            function=engineer_features,
            inputs={
                "source_feature_view": "preprocessed_customer_features"
            },
            outputs={
                "feature_view_name": "engineered_customer_features"
            },
            dependencies=["preprocess_data"],
            compute_pool_size="MEDIUM",
            memory_request="4Gi",
            cpu_request="2",
            timeout_minutes=60
        ),
        
        NodeConfig(
            name="create_train_test_split",
            description="Split engineered features into training and test sets",
            function=create_train_test_split,
            inputs={
                "source_feature_view": "engineered_customer_features",
                "test_size": 0.2,
                "random_seed": 42,
                "stratify_columns": ["churned"]
            },
            outputs={
                "train_feature_view": "train_customer_features",
                "test_feature_view": "test_customer_features"
            },
            dependencies=["engineer_features"],
            compute_pool_size="SMALL",
            memory_request="2Gi",
            cpu_request="1",
            timeout_minutes=30
        ),
        
        NodeConfig(
            name="train_model",
            description="Train XGBoost model for churn prediction",
            function=train_model,
            inputs={
                "train_feature_view": "train_customer_features",
                "target_column": "churned",
                "model_params": {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42
                },
                "cv_folds": 5,
                "eval_metric": "auc"
            },
            outputs={
                "model_name": "customer_churn_xgboost"
            },
            dependencies=["create_train_test_split"],
            compute_pool_size="LARGE",
            memory_request="8Gi",
            cpu_request="4",
            timeout_minutes=120
        ),
        
        NodeConfig(
            name="evaluate_model",
            description="Evaluate model performance and generate SHAP explanations",
            function=evaluate_model,
            inputs={
                "model_name": "customer_churn_xgboost",
                "test_feature_view": "test_customer_features",
                "target_column": "churned"
            },
            outputs={
                "evaluation_table": "MODEL_EVALUATION_RESULTS",
                "shap_plots_stage": "SHAP_PLOTS"
            },
            dependencies=["train_model"],
            compute_pool_size="MEDIUM",
            memory_request="4Gi",
            cpu_request="2",
            timeout_minutes=60
        )
    ]
    
    # Convert to dictionary format for compatibility
    pipeline_dicts = []
    for node in nodes:
        node_dict = {
            'name': node.name,
            'description': node.description,
            'function': node.function,
            'inputs': node.inputs,
            'outputs': node.outputs,
            'dependencies': node.dependencies,
            'compute_pool_size': node.compute_pool_size,
            'memory_request': node.memory_request,
            'cpu_request': node.cpu_request,
            'timeout_minutes': node.timeout_minutes
        }
        pipeline_dicts.append(node_dict)
    
    logger.info(f"Pipeline created with {len(pipeline_dicts)} nodes")
    return pipeline_dicts


def get_node_by_name(node_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific node configuration by name.
    
    Args:
        node_name: Name of the node to retrieve
    
    Returns:
        Node configuration dictionary or None if not found
    """
    pipeline = create_pipeline()
    
    for node in pipeline:
        if node['name'] == node_name:
            return node
    
    return None


def get_pipeline_dependencies() -> Dict[str, List[str]]:
    """
    Get the dependency mapping for the pipeline.
    
    Returns:
        Dictionary mapping node names to their dependencies
    """
    pipeline = create_pipeline()
    dependencies = {}
    
    for node in pipeline:
        dependencies[node['name']] = node['dependencies']
    
    return dependencies


def validate_pipeline() -> List[str]:
    """
    Validate the pipeline configuration.
    
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    try:
        pipeline = create_pipeline()
        dependencies = get_pipeline_dependencies()
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for dep in dependencies.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node_name in dependencies:
            if node_name not in visited:
                if has_cycle(node_name):
                    errors.append("Circular dependency detected in pipeline")
                    break
        
        # Check that all dependencies exist
        all_nodes = set(dependencies.keys())
        for node_name, deps in dependencies.items():
            for dep in deps:
                if dep not in all_nodes:
                    errors.append(f"Node '{node_name}' depends on non-existent node '{dep}'")
        
        # Check for duplicate node names
        node_names = [node['name'] for node in pipeline]
        if len(node_names) != len(set(node_names)):
            errors.append("Duplicate node names found in pipeline")
        
        # Validate node configurations
        for node in pipeline:
            # Check required fields
            required_fields = ['name', 'description', 'function', 'inputs', 'outputs']
            for field in required_fields:
                if field not in node or node[field] is None:
                    errors.append(f"Node '{node.get('name', 'unknown')}' missing required field: {field}")
            
            # Check function is callable
            if 'function' in node and not callable(node['function']):
                errors.append(f"Node '{node['name']}' function is not callable")
        
        # Warn if node imports not available
        if not NODE_IMPORTS_AVAILABLE:
            errors.append("Some node modules could not be imported - pipeline may not function correctly")
        
        logger.info(f"Pipeline validation completed. Found {len(errors)} errors.")
        
    except Exception as e:
        errors.append(f"Pipeline validation failed with exception: {str(e)}")
    
    return errors


def get_execution_order() -> List[str]:
    """
    Get the topological execution order of nodes.
    
    Returns:
        List of node names in execution order
    """
    dependencies = get_pipeline_dependencies()
    execution_order = []
    remaining_nodes = list(dependencies.keys())
    
    while remaining_nodes:
        # Find nodes with no unresolved dependencies
        ready_nodes = []
        for node in remaining_nodes:
            deps = dependencies[node]
            if all(dep in execution_order for dep in deps):
                ready_nodes.append(node)
        
        if not ready_nodes:
            logger.error("Cannot determine execution order - circular dependency or missing nodes")
            return []
        
        # Sort for consistent ordering
        ready_nodes.sort()
        
        for node in ready_nodes:
            execution_order.append(node)
            remaining_nodes.remove(node)
    
    return execution_order


if __name__ == "__main__":
    # Test pipeline creation and validation
    print("Testing pipeline configuration...")
    
    # Create pipeline
    pipeline = create_pipeline()
    print(f"Created pipeline with {len(pipeline)} nodes:")
    for node in pipeline:
        print(f"  - {node['name']}: {node['description']}")
    
    # Show dependencies
    deps = get_pipeline_dependencies()
    print(f"\nDependencies:")
    for node_name, node_deps in deps.items():
        if node_deps:
            print(f"  {node_name} → depends on: {', '.join(node_deps)}")
        else:
            print(f"  {node_name} → no dependencies (root node)")
    
    # Validate pipeline
    errors = validate_pipeline()
    if errors:
        print(f"\n❌ Pipeline validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ Pipeline validation passed!")
    
    # Show execution order
    order = get_execution_order()
    if order:
        print(f"\nExecution order: {' → '.join(order)}")
    else:
        print("\n❌ Could not determine execution order") 