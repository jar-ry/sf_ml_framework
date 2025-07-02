#!/usr/bin/env python3
"""
Generic node runner for MLOps framework.

This script serves as the container entrypoint and executes specific nodes
based on the --node argument passed to the container.
"""

import argparse
import logging
import sys
import os
from typing import Dict, Any, Optional
import traceback

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import create_pipeline
from snowflake_utils import get_feature_store, get_model_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLOps Framework Node Runner')
    parser.add_argument(
        '--node', 
        required=True, 
        help='Name of the node to execute'
    )
    parser.add_argument(
        '--inputs', 
        type=str,
        default='{}',
        help='JSON string of input parameters'
    )
    parser.add_argument(
        '--outputs', 
        type=str,
        default='{}',
        help='JSON string of output specifications'
    )
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to additional configuration file'
    )
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load additional configuration from file if provided.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not config_path or not os.path.exists(config_path):
        return {}
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def find_node_definition(node_name: str, pipeline_nodes: list) -> Optional[Dict[str, Any]]:
    """
    Find node definition in pipeline manifest.
    
    Args:
        node_name: Name of the node to find
        pipeline_nodes: List of node definitions from pipeline
        
    Returns:
        Node definition dictionary or None if not found
    """
    for node in pipeline_nodes:
        if node.get('name') == node_name:
            return node
    return None


def execute_node(node_definition: Dict[str, Any], inputs: Dict[str, Any], 
                outputs: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Execute a specific node function.
    
    Args:
        node_definition: Node definition from pipeline
        inputs: Input parameters
        outputs: Output specifications  
        config: Additional configuration
        
    Returns:
        True if execution successful, False otherwise
    """
    try:
        # Get the function to execute
        node_function = node_definition.get('function')
        if not node_function:
            logger.error(f"No function defined for node")
            return False
        
        # Initialize Snowflake services
        feature_store = get_feature_store()
        model_registry = get_model_registry()
        
        # Merge inputs with config
        combined_inputs = {**inputs, **config}
        
        # Log execution start
        logger.info(f"Starting execution of node: {node_definition.get('name')}")
        logger.info(f"Inputs: {combined_inputs}")
        logger.info(f"Outputs: {outputs}")
        
        # Execute the node function
        result = node_function(
            feature_store=feature_store,
            model_registry=model_registry,
            inputs=combined_inputs,
            outputs=outputs
        )
        
        logger.info(f"Node execution completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Node execution failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load additional config if provided
        config = load_config(args.config)
        
        # Parse inputs and outputs
        import json
        try:
            inputs = json.loads(args.inputs)
            outputs = json.loads(args.outputs)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse inputs/outputs JSON: {e}")
            sys.exit(1)
        
        # Get pipeline definition
        try:
            pipeline_nodes = create_pipeline()
        except Exception as e:
            logger.error(f"Failed to load pipeline definition: {e}")
            sys.exit(1)
        
        # Find the requested node
        node_definition = find_node_definition(args.node, pipeline_nodes)
        if not node_definition:
            logger.error(f"Node '{args.node}' not found in pipeline definition")
            logger.info(f"Available nodes: {[n.get('name') for n in pipeline_nodes]}")
            sys.exit(1)
        
        # Execute the node
        success = execute_node(node_definition, inputs, outputs, config)
        
        if success:
            logger.info("Node execution completed successfully")
            sys.exit(0)
        else:
            logger.error("Node execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 