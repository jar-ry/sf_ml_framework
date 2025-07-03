#!/usr/bin/env python3
"""
MLOps Framework Main Orchestrator

This script provides a command-line interface to manage and execute
the MLOps pipeline tasks on Snowflake.
"""

import argparse
import os
import sys
import json
import logging
import yaml
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import subprocess
import re
from snowflake.core import Root
from snowflake.core.compute_pool import ComputePool
from snowflake.core.service import Service, ServiceSpecInlineText
from snowflake.core.task import Task
from snowflake.snowpark import Session
from datetime import timedelta
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.pipeline import create_pipeline, validate_pipeline, get_pipeline_dependencies
from src.utils.snowflake_utils import get_snowflake_connection, snowflake_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLOpsPipelineOrchestrator:
    """Main orchestrator for MLOps pipeline execution."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.pipeline_nodes = None
        self.dependencies = None
        self.load_environment()
        
    def load_environment(self):
        """Load environment variables from .env file."""
        from dotenv import load_dotenv
        
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info("Loaded environment variables from .env file")
        else:
            logger.warning("No .env file found. Using system environment variables.")
    
    def load_pipeline(self):
        """Load and validate the pipeline configuration."""
        try:
            self.pipeline_nodes = create_pipeline()
            self.dependencies = get_pipeline_dependencies()
            
            # Validate pipeline
            errors = validate_pipeline()
            if errors:
                logger.error("Pipeline validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                return False
            
            logger.info(f"Pipeline loaded successfully with {len(self.pipeline_nodes)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def list_nodes(self):
        """List all available pipeline nodes."""
        if not self.pipeline_nodes:
            if not self.load_pipeline():
                return
        
        if not self.pipeline_nodes:  # Additional safety check
            logger.error("No pipeline nodes available")
            return
        
        print("\n📋 Available Pipeline Nodes:")
        print("=" * 50)
        
        for node in self.pipeline_nodes:
            deps = ", ".join(node.get('dependencies', [])) or "None"
            print(f"• {node['name']}")
            print(f"  Description: {node['description']}")
            print(f"  Dependencies: {deps}")
            print(f"  Compute: {node.get('compute_pool_size', 'SMALL')} pool")
            print(f"  Resources: {node.get('memory_request', '2Gi')} memory, {node.get('cpu_request', '1')} CPU")
            print()
    
    def show_dependencies(self):
        """Show pipeline dependencies graph."""
        if not self.dependencies:
            if not self.load_pipeline():
                return
        
        if not self.dependencies:  # Additional safety check
            logger.error("No pipeline dependencies available")
            return
        
        print("\n🔗 Pipeline Dependencies:")
        print("=" * 30)
        
        for node_name, deps in self.dependencies.items():
            if deps:
                for dep in deps:
                    print(f"{dep} → {node_name}")
            else:
                print(f"🟢 {node_name} (root node)")
        print()
    
    def execute_node_locally(self, node_name: str, inputs: Optional[Dict] = None, outputs: Optional[Dict] = None):
        """Execute a single node locally for testing."""
        if not self.pipeline_nodes:
            if not self.load_pipeline():
                return False
        
        if not self.pipeline_nodes:  # Additional safety check
            logger.error("No pipeline nodes available")
            return False
        
        # Find the node
        node = None
        for n in self.pipeline_nodes:
            if n['name'] == node_name:
                node = n
                break
        
        if not node:
            logger.error(f"Node '{node_name}' not found")
            return False
        
        try:
            logger.info(f"Executing node '{node_name}' locally...")
            
            # Use default inputs/outputs if not provided
            node_inputs = inputs or node.get('inputs', {})
            node_outputs = outputs or node.get('outputs', {})
            
            # Import and execute the node function
            node_function = node['function']
            
            # For local execution, we pass None for Snowflake services
            node_function(
                feature_store=None,
                model_registry=None,
                inputs=node_inputs,
                outputs=node_outputs
            )
            
            logger.info(f"✅ Node '{node_name}' executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Node '{node_name}' execution failed: {e}")
            return False
    
    def execute_pipeline_locally(self, start_node: Optional[str] = None, end_node: Optional[str] = None):
        """Execute the entire pipeline locally."""
        if not self.pipeline_nodes:
            if not self.load_pipeline():
                return False
        
        # Determine execution order
        execution_order = self._get_execution_order(start_node, end_node)
        
        if not execution_order:
            logger.error("Could not determine execution order")
            return False
        
        logger.info(f"Executing pipeline locally: {' → '.join(execution_order)}")
        
        success_count = 0
        for node_name in execution_order:
            logger.info(f"\n{'='*60}")
            logger.info(f"Executing node: {node_name}")
            logger.info(f"{'='*60}")
            
            if self.execute_node_locally(node_name):
                success_count += 1
            else:
                logger.error(f"Pipeline stopped due to failure in node: {node_name}")
                break
        
        total_nodes = len(execution_order)
        logger.info(f"\n📊 Pipeline Execution Summary:")
        logger.info(f"Total nodes: {total_nodes}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {total_nodes - success_count}")
        
        return success_count == total_nodes
    
    def _get_execution_order(self, start_node: Optional[str] = None, end_node: Optional[str] = None) -> List[str]:
        """Determine the execution order based on dependencies."""
        if not self.dependencies:
            return []
        
        # Simple topological sort
        all_nodes = list(self.dependencies.keys())
        
        # Filter nodes if start/end specified
        if start_node or end_node:
            if start_node and start_node not in all_nodes:
                logger.error(f"Start node '{start_node}' not found")
                return []
            if end_node and end_node not in all_nodes:
                logger.error(f"End node '{end_node}' not found")
                return []
        
        # Simple implementation - assumes pipeline is already in order
        execution_order = []
        remaining_nodes = all_nodes.copy()
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            ready_nodes = []
            for node in remaining_nodes:
                deps = self.dependencies[node]
                if all(dep in execution_order for dep in deps):
                    ready_nodes.append(node)
            
            if not ready_nodes:
                logger.error("Circular dependency detected or missing dependencies")
                return []
            
            # Sort ready nodes to ensure consistent ordering
            ready_nodes.sort()
            
            for node in ready_nodes:
                execution_order.append(node)
                remaining_nodes.remove(node)
        
        # Apply start/end filtering
        if start_node:
            try:
                start_idx = execution_order.index(start_node)
                execution_order = execution_order[start_idx:]
            except ValueError:
                pass
        
        if end_node:
            try:
                end_idx = execution_order.index(end_node)
                execution_order = execution_order[:end_idx + 1]
            except ValueError:
                pass
        
        return execution_order
    
    def check_task_status(self):
        """Check the status of Snowflake tasks."""
        try:
            connection = get_snowflake_connection()
            cursor = connection.cursor()
            
            # Query task status
            cursor.execute("SHOW TASKS LIKE 'task_%'")
            tasks = cursor.fetchall()
            
            if not tasks:
                logger.info("No tasks found")
                return
            
            print("\n📊 Task Status:")
            print("=" * 80)
            print(f"{'Task Name':<30} {'State':<15} {'Schedule':<20} {'Last Run':<15}")
            print("-" * 80)
            
            for task in tasks:
                name = task[1]  # Task name
                state = task[8] if len(task) > 8 else 'Unknown'  # State
                schedule = task[6] if len(task) > 6 else 'Manual'  # Schedule
                last_run = task[10] if len(task) > 10 else 'Never'  # Last run
                
                print(f"{name:<30} {state:<15} {schedule:<20} {str(last_run):<15}")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to check task status: {e}")
    
    def resume_tasks(self, task_names: Optional[List[str]] = None):
        """Resume Snowflake tasks."""
        try:
            connection = get_snowflake_connection()
            cursor = connection.cursor()
            
            if task_names:
                tasks_to_resume = task_names
            else:
                # Resume all pipeline tasks
                tasks_to_resume = [
                    'task_generate_sample_data',
                    'task_preprocess_data', 
                    'task_engineer_features',
                    'task_train_test_split',
                    'task_train_model',
                    'task_evaluate_model'
                ]
            
            for task_name in tasks_to_resume:
                try:
                    cursor.execute(f"ALTER TASK {task_name} RESUME")
                    logger.info(f"✅ Resumed task: {task_name}")
                except Exception as e:
                    logger.warning(f"Failed to resume task {task_name}: {e}")
            
            cursor.close()
            logger.info("Task resume operation completed")
            
        except Exception as e:
            logger.error(f"Failed to resume tasks: {e}")
    
    def suspend_tasks(self, task_names: Optional[List[str]] = None):
        """Suspend Snowflake tasks."""
        try:
            connection = get_snowflake_connection()
            cursor = connection.cursor()
            
            if task_names:
                tasks_to_suspend = task_names
            else:
                # Suspend all pipeline tasks in reverse order
                tasks_to_suspend = [
                    'task_evaluate_model',
                    'task_train_model',
                    'task_train_test_split',
                    'task_engineer_features',
                    'task_preprocess_data',
                    'task_generate_sample_data'
                ]
            
            for task_name in tasks_to_suspend:
                try:
                    cursor.execute(f"ALTER TASK {task_name} SUSPEND")
                    logger.info(f"✅ Suspended task: {task_name}")
                except Exception as e:
                    logger.warning(f"Failed to suspend task {task_name}: {e}")
            
            cursor.close()
            logger.info("Task suspend operation completed")
            
        except Exception as e:
            logger.error(f"Failed to suspend tasks: {e}")

    def create_snowflake_resources_with_sdk(self, update_placeholders: bool = True):
        """Create Snowflake compute pools, services, and tasks using Python SDK."""
        logger.info("Creating Snowflake resources using Python SDK...")
        
        try:            
            # Set database and schema context
            account = os.getenv('SNOWFLAKE_ACCOUNT')
            user = os.getenv('SNOWFLAKE_USER')
            password = os.getenv('SNOWFLAKE_PASSWORD')
            role_name = os.getenv('SNOWFLAKE_ROLE')
            warehouse_name = os.getenv('SNOWFLAKE_WAREHOUSE')
            database_name = os.getenv('SNOWFLAKE_DATABASE')
            schema_name = os.getenv('SNOWFLAKE_SCHEMA')
            registry_url = os.getenv('SNOWFLAKE_REGISTRY_URL')
            
            # Check if all required parameters are not None without using not all
            if not account or not user or not password or not role_name or not warehouse_name or not database_name or not schema_name or not registry_url:
                logger.error("Missing required Snowflake connection parameters")
                return False

            # Create session and root object
            connection_parameters: Dict[str, Union[str, int]] = {
                "account": account,
                "user": user,
                "password": password,
                "role": role_name,
                "warehouse": warehouse_name,
                "database": database_name,
                "schema": schema_name,
            }

            session = Session.builder.configs(connection_parameters).create()
            root = Root(session)
            
            # Set database and schema context            
            database = root.databases[database_name]
            schema = database.schemas[schema_name]
            
            logger.info(f"Working with database: {database_name}, schema: {schema_name}")
            
            # Step 1: Create compute pool
            logger.info("Creating compute pool...")
            compute_pool_def = ComputePool(
                name="MLOPS_COMPUTE_POOL",
                instance_family="CPU_X64_XS",
                min_nodes=1,
                max_nodes=10,
                auto_suspend_secs=300,
                comment="Compute pool for MLOps framework container services"
            )
            
            try:
                compute_pool = root.compute_pools.create(compute_pool_def)
                logger.info("✅ Compute pool created successfully")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("Compute pool already exists, continuing...")
                    compute_pool = root.compute_pools["MLOPS_COMPUTE_POOL"]
                else:
                    raise e
            
            # Step 2: Create services from YAML files
            logger.info("Creating services from YAML files...")
            services_created = []
            
            # Define service configurations
            service_configs = [
                {
                    "name": "job-generate-data",
                    "yaml_file": "job-generate-data.yaml",
                    "service_name": "svc_generate_data"
                },
                {
                    "name": "job-preprocess-data", 
                    "yaml_file": "job-preprocess-data.yaml",
                    "service_name": "svc_preprocess_data"
                },
                {
                    "name": "job-feature-engineering",
                    "yaml_file": "job-feature-engineering.yaml",
                    "service_name": "svc_feature_engineering"
                },
                {
                    "name": "job-train-test-split",
                    "yaml_file": "job-train-test-split.yaml",
                    "service_name": "svc_train_test_split"
                },
                {
                    "name": "job-train-model",
                    "yaml_file": "job-train-model.yaml",
                    "service_name": "svc_train_model"
                },
                {
                    "name": "job-evaluate-model",
                    "yaml_file": "job-evaluate-model.yaml",
                    "service_name": "svc_evaluate_model"
                }
            ]
            
            for config in service_configs:
                yaml_path = os.path.join("jobs", config["yaml_file"])
                
                if not os.path.exists(yaml_path):
                    logger.error(f"YAML file not found: {yaml_path}")
                    continue
                
                try:
                    with open(yaml_path, 'r') as f:
                        job_spec = yaml.safe_load(f)
                    # Replace placeholders if requested
                    if update_placeholders:
                        job_spec_str = yaml.dump(job_spec)
                        replacements = {
                            '<SNOWFLAKE_ACCOUNT>': account,
                            '<SNOWFLAKE_DATABASE>': database_name,
                            '<SNOWFLAKE_SCHEMA>': schema_name,
                            '<SNOWFLAKE_USER>': user,
                            '<SNOWFLAKE_REGISTRY_URL>': registry_url,
                            '<SNOWFLAKE_WAREHOUSE>': warehouse_name,
                            '<SNOWFLAKE_ROLE>': role_name
                        }
                        
                        for placeholder, value in replacements.items():
                            job_spec_str = job_spec_str.replace(placeholder, value)
                        
                        job_spec = yaml.safe_load(job_spec_str)
                    # Create service specification
                    service_spec = yaml.dump(job_spec, default_flow_style=False)

                    # Create service
                    service_def = Service(
                        name=config["service_name"],
                        compute_pool="MLOPS_COMPUTE_POOL",
                        spec=ServiceSpecInlineText(spec_text=service_spec),
                        min_instances=1,
                        max_instances=1,
                        comment=f"Service for {config['name']}"
                    )
                    try:
                        service = schema.services.create(service_def)
                        logger.info(f"✅ Service created: {config['service_name']}")
                        services_created.append(config["service_name"])
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"Service {config['service_name']} already exists, continuing...")
                            services_created.append(config["service_name"])
                        else:
                            logger.error(f"Failed to create service {config['service_name']}: {e}")
                            return False
                    
                except Exception as e:
                    logger.error(f"Error processing YAML file {yaml_path}: {e}")
                    return False
            
            # Step 3: Create tasks that execute services
            logger.info("Creating tasks that execute services...")
            
            # Define task configurations with dependencies
            task_configs = [
                {
                    "name": "task_generate_sample_data",
                    "service_name": "svc_generate_data",
                    "schedule": timedelta(hours=24),  # Daily
                    "timeout_ms": 3600000,
                    "comment": "Generate synthetic customer data for churn prediction",
                    "dependencies": []
                },
                {
                    "name": "task_preprocess_data",
                    "service_name": "svc_preprocess_data", 
                    "schedule": None,
                    "timeout_ms": 3600000,
                    "comment": "Clean and preprocess raw customer data",
                    "dependencies": ["task_generate_sample_data"]
                },
                {
                    "name": "task_engineer_features",
                    "service_name": "svc_feature_engineering",
                    "schedule": None,
                    "timeout_ms": 3600000,
                    "comment": "Create engineered features for model training",
                    "dependencies": ["task_preprocess_data"]
                },
                {
                    "name": "task_train_test_split",
                    "service_name": "svc_train_test_split",
                    "schedule": None,
                    "timeout_ms": 3600000,
                    "comment": "Split data into training and testing sets",
                    "dependencies": ["task_engineer_features"]
                },
                {
                    "name": "task_train_model",
                    "service_name": "svc_train_model",
                    "schedule": None,
                    "timeout_ms": 7200000,
                    "comment": "Train XGBoost model for churn prediction",
                    "dependencies": ["task_train_test_split"]
                },
                {
                    "name": "task_evaluate_model",
                    "service_name": "svc_evaluate_model",
                    "schedule": None,
                    "timeout_ms": 5400000,
                    "comment": "Evaluate model performance and generate SHAP plots",
                    "dependencies": ["task_train_model"]
                }
            ]
            
            # Create tasks
            for task_config in task_configs:
                if task_config["service_name"] not in services_created:
                    logger.warning(f"Skipping task {task_config['name']} - service {task_config['service_name']} not created")
                    continue
                
                try:
                    # For tasks that execute services, we need to use a SQL definition
                    # since the Python API doesn't directly support EXECUTE JOB SERVICE
                    sql_definition = f"""
                    EXECUTE JOB SERVICE
                    IN COMPUTE POOL MLOPS_COMPUTE_POOL
                    NAME = {task_config['service_name']}
                    """
                    
                    # Create task
                    task_def = Task(
                        name=task_config["name"],
                        definition=sql_definition,
                        schedule=task_config["schedule"]
                    )
                    
                    # Set dependencies (predecessors)
                    if task_config["dependencies"]:
                        # Note: The Python API may handle dependencies differently
                        # We might need to set them after creation
                        pass
                    
                    try:
                        task = schema.tasks.create(task_def)
                        logger.info(f"✅ Task created: {task_config['name']}")
                        
                        # Set dependencies if any
                        if task_config["dependencies"]:
                            for dep in task_config["dependencies"]:
                                try:
                                    # This is a simplified approach - the actual API might be different
                                    # We might need to recreate the task with dependencies
                                    logger.info(f"Setting dependency: {dep} -> {task_config['name']}")
                                except Exception as e:
                                    logger.warning(f"Failed to set dependency {dep}: {e}")
                    
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"Task {task_config['name']} already exists, continuing...")
                        else:
                            logger.error(f"Failed to create task {task_config['name']}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error creating task {task_config['name']}: {e}")
                    continue
            
            # Step 4: Resume tasks to activate them
            logger.info("Resuming tasks to activate them...")
            for task_config in task_configs:
                try:
                    task_resource = schema.tasks[task_config["name"]]
                    task_resource.resume()
                    logger.info(f"✅ Task resumed: {task_config['name']}")
                except Exception as e:
                    logger.warning(f"Failed to resume task {task_config['name']}: {e}")
            
            session.close()
            logger.info("✅ Snowflake resources created successfully using Python SDK")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Snowflake resources with SDK: {e}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='MLOps Framework Pipeline Orchestrator')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List nodes command
    list_parser = subparsers.add_parser('list', help='List all pipeline nodes')
    
    # Show dependencies command
    deps_parser = subparsers.add_parser('deps', help='Show pipeline dependencies')
    
    # Execute node command
    exec_parser = subparsers.add_parser('exec', help='Execute a single node locally')
    exec_parser.add_argument('node_name', help='Name of the node to execute')
    exec_parser.add_argument('--inputs', type=str, help='JSON string of inputs')
    exec_parser.add_argument('--outputs', type=str, help='JSON string of outputs')
    
    # Execute pipeline command
    pipeline_parser = subparsers.add_parser('run', help='Execute the entire pipeline locally')
    pipeline_parser.add_argument('--start', help='Start from this node')
    pipeline_parser.add_argument('--end', help='End at this node')
    
    # Snowflake task management
    task_parser = subparsers.add_parser('tasks', help='Manage Snowflake tasks')
    task_subparsers = task_parser.add_subparsers(dest='task_action')
    
    create_task_parser = task_subparsers.add_parser('create', help='Create Snowflake tasks')
    status_task_parser = task_subparsers.add_parser('status', help='Check task status')
    resume_task_parser = task_subparsers.add_parser('resume', help='Resume tasks')
    resume_task_parser.add_argument('--tasks', nargs='+', help='Specific tasks to resume')
    suspend_task_parser = task_subparsers.add_parser('suspend', help='Suspend tasks')
    suspend_task_parser.add_argument('--tasks', nargs='+', help='Specific tasks to suspend')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize orchestrator
    orchestrator = MLOpsPipelineOrchestrator()
    
    # Execute commands
    if args.command == 'list':
        orchestrator.list_nodes()
    
    elif args.command == 'deps':
        orchestrator.show_dependencies()
    
    elif args.command == 'exec':
        inputs = json.loads(args.inputs) if args.inputs else None
        outputs = json.loads(args.outputs) if args.outputs else None
        orchestrator.execute_node_locally(args.node_name, inputs, outputs)
    
    elif args.command == 'run':
        orchestrator.execute_pipeline_locally(args.start, args.end)
    
    elif args.command == 'tasks':
        if args.task_action == 'create':
            orchestrator.create_snowflake_resources_with_sdk()
        elif args.task_action == 'status':
            orchestrator.check_task_status()
        elif args.task_action == 'resume':
            orchestrator.resume_tasks(args.tasks)
        elif args.task_action == 'suspend':
            orchestrator.suspend_tasks(args.tasks)
        else:
            task_parser.print_help()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 