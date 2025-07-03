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
from typing import List, Dict, Any, Optional
from datetime import datetime
import subprocess
import re
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
    
    def create_snowflake_tasks(self, update_placeholders: bool = True):
        """Create or update Snowflake tasks."""
        logger.info("Creating Snowflake tasks...")
        
        sql_file = "tasks/create_pipeline_tasks.sql"
        if not os.path.exists(sql_file):
            logger.error(f"SQL file not found: {sql_file}")
            return False
        
        # Read SQL file
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        if update_placeholders:
            # Replace placeholders with environment variables
            replacements = {
                '<SNOWFLAKE_ACCOUNT>': os.getenv('SNOWFLAKE_ACCOUNT', None),
                '<SNOWFLAKE_DATABASE>': os.getenv('SNOWFLAKE_DATABASE', None),
                '<SNOWFLAKE_SCHEMA>': os.getenv('SNOWFLAKE_SCHEMA', None),
                '<SNOWFLAKE_USER>': os.getenv('SNOWFLAKE_USER', None),
                '<SNOWFLAKE_REGISTRY_URL>': os.getenv('SNOWFLAKE_REGISTRY_URL', None)
            }
            
            for placeholder, value in replacements.items():
                if value is not None:
                    sql_content = sql_content.replace(placeholder, value)
                else:
                    logger.error(f"Environment variable {placeholder} is not set")
                    return False
            
            # Save updated SQL
            updated_sql_file = "tasks/create_pipeline_tasks_configured.sql"
            with open(updated_sql_file, 'w') as f:
                f.write(sql_content)
            
            logger.info(f"SQL file updated and saved as: {updated_sql_file}")
            sql_file = updated_sql_file
        
        try:
            # Execute SQL using Snowflake connection
            connection = get_snowflake_connection()
            cursor = connection.cursor()
            
            # Split and execute SQL statements
            statements = sql_content.split(';')
            for statement in statements:
                statement = statement.strip()
                # Strip comments lines but keep following lines within the statement
                statement = re.sub(r'--.*', '', statement)
                print(statement)
                if statement and not statement.startswith('--'):
                    cursor.execute(statement)
            
            logger.info("✅ Snowflake tasks created successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Snowflake tasks: {e}")
            return False
    
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
            orchestrator.create_snowflake_tasks()
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