# MLOps Framework Usage Guide

This guide shows how to use the main.py orchestrator to manage your MLOps pipeline.

## Prerequisites

1. **Environment Setup**: 
   
   **Option A: Using Conda (Recommended)**
   ```bash
   # Install conda/miniconda if you haven't already
   # https://docs.conda.io/en/latest/miniconda.html
   
   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate mlops-framework
   
   # Configure credentials
   cp config.env .env
   # Edit .env with your Snowflake account details
   ```

   **Option B: Using pip**
   ```bash
   # Make sure you have Python 3.10+
   python --version
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure credentials
   cp config.env .env
   # Edit .env with your Snowflake account details
   ```

## Main Commands

### 1. List Available Nodes
```bash
python main.py list
```

Shows all pipeline nodes with their descriptions, dependencies, and resource requirements.

### 2. Show Pipeline Dependencies
```bash
python main.py deps
```

Displays the dependency graph showing which nodes depend on others.

### 3. Execute Single Node Locally
```bash
# Run a specific node
python main.py exec generate_sample_data

# Run with custom inputs/outputs
python main.py exec preprocess_data --inputs '{"source_feature_view": "custom_data"}' --outputs '{"feature_view_name": "my_processed_data"}'
```

### 4. Execute Full Pipeline Locally
```bash
# Run entire pipeline
python main.py run

# Run from specific node to end
python main.py run --start engineer_features

# Run up to specific node
python main.py run --end train_model

# Run specific subset
python main.py run --start preprocess_data --end train_model
```

### 5. Manage Snowflake Tasks

#### Create Tasks
```bash
python main.py tasks create
```

Creates the Snowflake Task DAG with all dependencies and schedules.

#### Check Task Status
```bash
python main.py tasks status
```

Shows current status of all pipeline tasks.

#### Resume Tasks
```bash
# Resume all tasks
python main.py tasks resume

# Resume specific tasks
python main.py tasks resume --tasks task_generate_sample_data task_preprocess_data
```

#### Suspend Tasks
```bash
# Suspend all tasks
python main.py tasks suspend

# Suspend specific tasks
python main.py tasks suspend --tasks task_train_model task_evaluate_model
```

## Example Workflows

### Local Development and Testing

1. **Setup environment**:
   ```bash
   # Using conda
   conda activate mlops-framework
   
   # OR ensure pip environment is active
   ```

2. **Test individual nodes**:
   ```bash
   python main.py exec generate_sample_data
   python main.py exec preprocess_data
   ```

3. **Test full pipeline locally**:
   ```bash
   python main.py run
   ```

4. **Debug specific pipeline segment**:
   ```bash
   python main.py run --start engineer_features --end train_model
   ```

### Production Deployment

1. **Deploy to Snowflake**:
   ```bash
   # Create tasks
   python main.py tasks create
   
   # Check status
   python main.py tasks status
   
   # Resume execution
   python main.py tasks resume
   ```

2. **Monitor pipeline**:
   ```bash
   # Check task status regularly
   python main.py tasks status
   ```

3. **Pause/Resume as needed**:
   ```bash
   # Pause for maintenance
   python main.py tasks suspend
   
   # Resume after maintenance
   python main.py tasks resume
   ```

## Pipeline Nodes

The customer churn prediction pipeline consists of:

1. **generate_sample_data** - Creates synthetic customer data
2. **preprocess_data** - Cleans and validates the data
3. **engineer_features** - Creates advanced features for modeling
4. **create_train_test_split** - Splits data into training and test sets
5. **train_model** - Trains XGBoost model with cross-validation
6. **evaluate_model** - Evaluates model and generates SHAP explanations

## Configuration

### Environment Variables (config.env)

Key variables to configure:
- `SNOWFLAKE_ACCOUNT` - Your Snowflake account URL
- `SNOWFLAKE_USER` - Username for Snowflake
- `SNOWFLAKE_PASSWORD` - Password (or use key-pair auth)
- `SNOWFLAKE_DATABASE` - Database for the pipeline
- `SNOWFLAKE_SCHEMA` - Schema for the pipeline
- `SNOWFLAKE_WAREHOUSE` - Warehouse for task execution

### Resource Configuration

Node resources are defined in `src/utils/pipeline.py`:
- **SMALL**: 1-2 CPU, 1-2Gi memory
- **MEDIUM**: 2-4 CPU, 4Gi memory  
- **LARGE**: 4+ CPU, 8Gi memory

## Environment Management

### Managing Conda Environment

```bash
# List environments
conda env list

# Activate environment
conda activate mlops-framework

# Deactivate environment
conda deactivate

# Update environment from yml file
conda env update -f environment.yml

# Remove environment
conda env remove -n mlops-framework
```

### Adding New Dependencies

**For conda:**
```bash
# Add to environment.yml, then update
conda env update -f environment.yml

# OR install directly (temporary)
conda install package-name
```

**For pip:**
```bash
# Add to requirements.txt, then install
pip install -r requirements.txt

# OR install directly
pip install package-name
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure environment is activated
   ```bash
   # For conda
   conda activate mlops-framework
   
   # For pip, ensure dependencies are installed
   pip install -r requirements.txt
   ```

2. **Python version issues**: Check Python version
   ```bash
   python --version  # Should be 3.10+
   ```

3. **Connection errors**: Check Snowflake credentials in `.env`

4. **Task creation fails**: Ensure you have proper permissions in Snowflake

5. **Node execution fails**: Check logs for specific error messages

### Environment Issues

1. **Conda environment not found**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Package conflicts**:
   ```bash
   # Remove and recreate environment
   conda env remove -n mlops-framework
   conda env create -f environment.yml
   ```

3. **Mixed conda/pip issues**:
   - Prefer conda packages when available
   - Use pip only for packages not in conda-forge

### Logs and Debugging

- Local execution logs appear in console
- Snowflake task logs are in Snowflake's task history
- Set `LOG_LEVEL=DEBUG` in `.env` for verbose logging

### Getting Help

Run any command with `--help` to see available options:
```bash
python main.py --help
python main.py tasks --help
``` 