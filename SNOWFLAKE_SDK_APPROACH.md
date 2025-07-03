# Using Snowflake Python SDK for MLOps Framework

This document explains how to use the Snowflake Python SDK instead of executing SQL to create compute pools, services, and tasks.

## Prerequisites

Install the required Snowflake packages:

```bash
pip install snowflake-connector-python[pandas]
pip install snowflake-snowpark-python
# Note: snowflake-core may be included in newer versions of snowflake-snowpark-python
```

## Architecture Overview

The SDK approach provides several advantages:

✅ **Type-safe Python objects** - No SQL string manipulation  
✅ **Better error handling** - Native Python exceptions  
✅ **IDE support** - Autocomplete and type checking  
✅ **Programmatic control** - Easy resource lifecycle management  
✅ **YAML integration** - Direct use of job YAML files as service specs  

## Implementation Strategy

### 1. Session and Connection Setup

```python
from snowflake.core import Root
from snowflake.snowpark import Session

# Create session using connection parameters
session = Session.builder.config("connection_name", "snowflake").create()
root = Root(session)

# Set context
database = root.databases[database_name]
schema = database.schemas[schema_name]
```

### 2. Compute Pool Creation

```python
from snowflake.core.compute_pool import ComputePool

compute_pool_def = ComputePool(
    name="MLOPS_COMPUTE_POOL",
    instance_family="CPU_X64_XS",
    min_nodes=1,
    max_nodes=10,
    auto_suspend_secs=300,
    comment="Compute pool for MLOps framework container services"
)

# Create or get existing
try:
    compute_pool = root.compute_pools.create(compute_pool_def)
except Exception as e:
    if "already exists" in str(e).lower():
        compute_pool = root.compute_pools["MLOPS_COMPUTE_POOL"]
    else:
        raise e
```

### 3. Service Creation from YAML Files

```python
from snowflake.core.service import Service, ServiceSpecInlineText
import yaml

# Read job YAML file
with open("jobs/job-generate-data.yaml", 'r') as f:
    job_spec = yaml.safe_load(f)

# Replace placeholders
job_spec_str = yaml.dump(job_spec, default_flow_style=False)
replacements = {
    '<SNOWFLAKE_ACCOUNT>': os.getenv('SNOWFLAKE_ACCOUNT'),
    '<SNOWFLAKE_DATABASE>': os.getenv('SNOWFLAKE_DATABASE'),
    '<SNOWFLAKE_SCHEMA>': os.getenv('SNOWFLAKE_SCHEMA'),
    '<SNOWFLAKE_USER>': os.getenv('SNOWFLAKE_USER'),
    '<SNOWFLAKE_REGISTRY_URL>': os.getenv('SNOWFLAKE_REGISTRY_URL')
}

for placeholder, value in replacements.items():
    job_spec_str = job_spec_str.replace(placeholder, value)

job_spec = yaml.safe_load(job_spec_str)

# Create service
service_spec = yaml.dump(job_spec, default_flow_style=False)

service_def = Service(
    name="svc_generate_data",
    compute_pool="MLOPS_COMPUTE_POOL",
    spec=ServiceSpecInlineText(service_spec),
    min_instances=1,
    max_instances=1,
    comment="Service for job-generate-data"
)

service = schema.services.create(service_def)
```

### 4. Task Creation for Service Execution

```python
from snowflake.core.task import Task
from datetime import timedelta

# Create task that executes the service
sql_definition = """
EXECUTE JOB SERVICE
IN COMPUTE POOL MLOPS_COMPUTE_POOL
NAME = svc_generate_data
"""

task_def = Task(
    name="task_generate_sample_data",
    definition=sql_definition,
    schedule=timedelta(hours=24)  # Daily schedule
)

task = schema.tasks.create(task_def)
```

### 5. Task Dependency Management

```python
# For tasks with dependencies, you may need to recreate them
# or use the create_or_alter method

# Example: Create dependent task
dependent_task_def = Task(
    name="task_preprocess_data",
    definition="EXECUTE JOB SERVICE IN COMPUTE POOL MLOPS_COMPUTE_POOL NAME = svc_preprocess_data",
    # Dependencies may need to be set via SQL or separate API calls
)

dependent_task = schema.tasks.create(dependent_task_def)

# Set predecessor relationship (API may vary)
# This might require SQL execution or specific API calls
```

### 6. Pipeline Activation

```python
# Resume tasks to activate them
task_names = [
    "task_generate_sample_data",
    "task_preprocess_data", 
    "task_engineer_features",
    "task_train_test_split",
    "task_train_model",
    "task_evaluate_model"
]

for task_name in task_names:
    try:
        task_resource = schema.tasks[task_name]
        task_resource.resume()
        print(f"✅ Task resumed: {task_name}")
    except Exception as e:
        print(f"Failed to resume task {task_name}: {e}")
```

## Complete Workflow

1. **Setup**: Create session and root object
2. **Compute Pool**: Create compute pool using ComputePool API
3. **Services**: Read YAML files and create services using Service API
4. **Tasks**: Create tasks that execute services using Task API
5. **Dependencies**: Set task dependencies (may require SQL)
6. **Activation**: Resume tasks to start the pipeline

## Benefits Over SQL Approach

### Type Safety
```python
# SDK approach - type-safe
compute_pool = ComputePool(name="POOL", instance_family="CPU_X64_XS")

# vs SQL approach - string manipulation
sql = f"CREATE COMPUTE POOL {pool_name} INSTANCE_FAMILY = {family}"
```

### Error Handling
```python
# SDK approach - native exceptions
try:
    service = schema.services.create(service_def)
except ServiceAlreadyExistsException:
    service = schema.services[service_name]

# vs SQL approach - parsing error messages
if "already exists" in str(error).lower():
    # Handle error
```

### Resource Management
```python
# SDK approach - direct object manipulation
service.suspend()
service.resume()
service.drop()

# vs SQL approach - SQL string generation
cursor.execute(f"ALTER SERVICE {service_name} SUSPEND")
```

## Integration with Existing Code

To integrate the SDK approach with the existing `main.py`:

1. **Install packages**: Add SDK packages to `requirements.txt`
2. **Add method**: Use the `create_snowflake_resources_with_sdk()` method
3. **CLI command**: Use `python main.py tasks create-sdk`
4. **Configuration**: Ensure proper environment variables are set

## Usage Example

```bash
# Set environment variables
export SNOWFLAKE_ACCOUNT=your_account
export SNOWFLAKE_DATABASE=MLOPS_DB
export SNOWFLAKE_SCHEMA=MLOPS_SCHEMA
export SNOWFLAKE_USER=your_user
export SNOWFLAKE_REGISTRY_URL=your_registry_url

# Create resources using SDK
python main.py tasks create-sdk

# Check status
python main.py tasks status

# Manage tasks
python main.py tasks resume
python main.py tasks suspend
```

## Troubleshooting

### Module Import Errors
If you get `No module named 'snowflake.core'`:
```bash
pip install --upgrade snowflake-snowpark-python
# Or try the specific package
pip install snowflake-core
```

### Connection Issues
Ensure your Snowflake connection parameters are properly configured in your environment or configuration file.

### API Parameter Errors
Some API parameters might differ from documentation. Check the latest API reference and adjust accordingly.

## Future Enhancements

1. **DAG API**: Use the DAG API for more complex task graphs
2. **Service Functions**: Integrate with Snowflake service functions
3. **Model Registry**: Use SDK for ML model management
4. **Monitoring**: Add SDK-based monitoring and alerting

This SDK approach provides a more maintainable and type-safe way to manage your MLOps infrastructure on Snowflake.

# Container Session Inheritance Implementation

## Overview
Container Session Inheritance is now implemented as the **recommended authentication approach** for the MLOps framework. This approach provides the highest security and simplicity by allowing containers running inside Snowflake to inherit the execution context rather than managing separate credentials.

## How It Works

### Architecture
```
Snowflake Task → Container Service → Container → Node Execution
     ↓              ↓                    ↓            ↓
Session Context → Inherited Session → No Auth → Authenticated APIs
```

### Authentication Flow
1. **Snowflake Task Execution**: When a Snowflake Task executes, it runs within an authenticated Snowflake session
2. **Container Launch**: The Task launches a Container Service with the same session context
3. **Session Inheritance**: The container inherits the parent session's authentication context
4. **No Explicit Auth**: No usernames, passwords, or keys needed in the container
5. **Native API Access**: Feature Store and Model Registry APIs work seamlessly

## Implementation Details

### Container Detection (`snowflake_utils.py`)
The framework automatically detects when it's running in a container:

```python
def _detect_container_mode(self) -> bool:
    """Detect if we're running inside a Snowflake container."""
    
    # Check for Snowflake container environment indicators
    container_indicators = [
        'SNOWFLAKE_CONTAINER_RUNTIME',  # Snowflake container runtime
        'SNOWFLAKE_SERVICE_NAME',       # Service name when running as ML Job
        'SNOWFLAKE_TASK_NAME'           # Task name when executing via Task
    ]
    
    for indicator in container_indicators:
        if os.getenv(indicator):
            return True
    
    # Check if credentials are not provided (typical container scenario)
    if (os.getenv('SNOWFLAKE_ACCOUNT') and 
        not os.getenv('SNOWFLAKE_PASSWORD') and 
        not os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH')):
        return True
    
    return False
```

### Session Inheritance Logic
When container mode is detected, the framework uses the current session context:

```python
if self._is_container_mode:
    # Container Session Inheritance mode
    try:
        from snowflake.snowpark import Session
        
        # Get the current session context (already authenticated)
        session = Session.builder.getOrCreate()
        
        # Use session directly for Feature Store and Model Registry
        self._feature_store = FeatureStore(
            session=session,
            database=database,
            name=schema,
            default_warehouse=warehouse
        )
        
        logger.info("✅ Using Container Session Inheritance")
        
    except Exception as e:
        logger.warning(f"Container session inheritance failed: {e}")
        # Fallback to explicit authentication
```

### Fallback Strategy
If container session inheritance fails, the framework automatically falls back to explicit authentication:

```python
except Exception as e:
    logger.warning(f"Failed to use container session inheritance: {e}")
    logger.info("Falling back to explicit authentication")
    # Create explicit connection with credentials
    connection = self._create_explicit_connection()
```

## Security Benefits

### ✅ Advantages of Container Session Inheritance

1. **No Credential Storage**: No need to store usernames, passwords, or private keys
2. **No Credential Transmission**: No secrets passed to containers
3. **Automatic Rotation**: Session tokens rotate automatically
4. **Principle of Least Privilege**: Inherits only the permissions of the executing context
5. **Audit Trail**: All operations traced back to the original session
6. **Simplified Management**: No credential lifecycle management

### ❌ What We Eliminated

1. **Hardcoded Credentials**: No secrets in code or configuration
2. **Environment Variables**: No `SNOWFLAKE_PASSWORD` needed in containers
3. **Key Management**: No private key distribution
4. **Credential Rotation**: No manual key rotation processes
5. **Secret Storage**: No external secret management complexity

## Environment Configuration

### Container Environment (Production)
```yaml
# job-*.yaml files - NO CREDENTIALS NEEDED
spec:
  containers:
    - name: node-container
      image: <registry>/mlops-framework:latest
      args: ["--node", "generate_data"]
      env:
        - name: SNOWFLAKE_ACCOUNT
          value: "<account>"
        - name: SNOWFLAKE_DATABASE  
          value: "<database>"
        - name: SNOWFLAKE_SCHEMA
          value: "<schema>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "<warehouse>"
        # NO PASSWORD OR KEYS NEEDED!
```

### Local Environment (.env)
```bash
# For local development only
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password  # OR use private key
SNOWFLAKE_DATABASE=your-database
SNOWFLAKE_SCHEMA=your-schema
SNOWFLAKE_WAREHOUSE=your-warehouse
```

## Node Implementation

### Standard Node Pattern
Nodes don't need to change - they receive authenticated objects:

```python
def generate_sample_data(feature_store, model_registry, inputs: Dict[str, Any], outputs: Dict[str, Any]):
    """
    Node function - receives authenticated objects automatically.
    
    Args:
        feature_store: Already authenticated FeatureStore instance
        model_registry: Already authenticated Registry instance
        inputs: Input parameters
        outputs: Output specifications
    """
    # Detect execution mode
    is_local_mode = feature_store is None
    
    if is_local_mode:
        # Local execution - save to files
        logger.info("Running in LOCAL mode")
        # ... save to data/ folder
    else:
        # Container execution - use authenticated services
        logger.info("Running in CONTAINER mode with inherited session")
        # ... use feature_store and model_registry directly
```

## Execution Flow

### 1. Task Execution
```sql
-- Task executes with authenticated session
CREATE TASK task_generate_data
  WAREHOUSE = compute_wh
  SCHEDULE = 'USING CRON 0 2 * * * UTC'
AS
  EXECUTE JOB SERVICE
  IN COMPUTE POOL mlops_compute_pool
  NAME = svc_generate_data;
```

### 2. Container Launch
```yaml
# Service launches container with inherited session
spec:
  containers:
    - name: node-container
      image: <registry>/mlops-framework:latest
      args: ["--node", "generate_data"]
      # Session context inherited automatically
```

### 3. Node Execution
```python
# run_node.py automatically detects container mode
feature_store = get_feature_store()  # Uses inherited session
model_registry = get_model_registry()  # Uses inherited session

# Passes authenticated objects to node
node_function(feature_store, model_registry, inputs, outputs)
```

## Testing

### Local Testing
```bash
# Test with explicit credentials
python main.py exec generate_data --inputs '{"customer_count": 1000}'
```

### Container Testing
```bash
# Test container session inheritance
python main.py tasks create-sdk  # Creates containers
# Tasks execute with inherited sessions automatically
```

## Troubleshooting

### Common Issues

**Container mode not detected**:
```bash
# Check for container environment indicators
echo $SNOWFLAKE_CONTAINER_RUNTIME
echo $SNOWFLAKE_SERVICE_NAME
echo $SNOWFLAKE_TASK_NAME
```

**Session inheritance fails**:
```bash
# Check logs for fallback to explicit authentication
tail -f /var/log/container.log | grep "Container Session Inheritance"
```

**Permission errors**:
```bash
# Verify task execution role has required permissions
SHOW GRANTS TO ROLE mlops_role;
```

### Debug Commands

```python
# Test container detection
from src.utils.snowflake_utils import SnowflakeManager
manager = SnowflakeManager()
print(f"Container mode: {manager._is_container_mode}")

# Test session inheritance
feature_store = manager.get_feature_store()
print(f"Feature Store initialized: {feature_store is not None}")
```

## Best Practices

### Security
1. **Use Role-Based Access Control**: Assign minimal required permissions to task execution roles
2. **Audit Regularly**: Monitor task execution logs for authentication events
3. **Rotate Regularly**: While session inheritance handles this automatically, review task permissions periodically

### Performance
1. **Session Reuse**: The framework reuses sessions across node executions
2. **Connection Pooling**: Snowflake handles connection pooling internally
3. **Resource Management**: Sessions are cleaned up automatically when containers terminate

### Monitoring
1. **Log Authentication Mode**: Monitor logs to confirm container vs local mode
2. **Track Session Usage**: Monitor session duration and resource usage
3. **Alert on Fallbacks**: Set up alerts if explicit authentication fallbacks occur frequently

## Migration Guide

### From Password Authentication
1. Remove `SNOWFLAKE_PASSWORD` from container environment
2. Deploy updated container image with session inheritance
3. Test container execution
4. Monitor logs for successful session inheritance

### From Key-Pair Authentication
1. Remove `SNOWFLAKE_PRIVATE_KEY_PATH` from container environment
2. Keep key-pair setup for local development
3. Deploy updated container image
4. Verify container uses session inheritance while local uses keys

## Summary

Container Session Inheritance provides the most secure and maintainable authentication approach for the MLOps framework:

- **🔐 Maximum Security**: No credential storage or transmission
- **🚀 Simplified Operations**: No credential management overhead  
- **🛡️ Automatic Rotation**: Session tokens handled by Snowflake
- **🔍 Full Audit Trail**: All operations traced to original session
- **🔄 Backward Compatible**: Local development still uses explicit authentication

This approach aligns with Snowflake's native security model and provides the best foundation for production MLOps deployments. 