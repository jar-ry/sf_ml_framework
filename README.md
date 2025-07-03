# MLOps Framework on Snowflake ❄️

A modular, resilient, and scalable MLOps framework built natively on Snowflake, treating each data science pipeline step as an independent, containerized ML Job orchestrated by Snowflake Tasks.

## 🏗️ Architecture Overview

This framework leverages Snowflake's native MLOps capabilities:

- **Orchestration**: Snowflake Tasks form DAG structure
- **Compute**: Snowpark Container Services with Compute Pools  
- **Storage**: Feature Store for features, Model Registry for models
- **Registry**: Snowflake Image Registry (no external dependencies)

```
Git Repo (Python Code) → CI/CD Pipeline → Docker Image → Snowflake Image Registry
                                                              ↓
Snowflake Tasks (DAG) → ML Jobs → Container Execution → Feature Store/Model Registry
```

## 📁 Project Structure

```
sf_ml_framework/
├── src/
│   ├── nodes/                    # Pipeline nodes (each is a containerized step)
│   │   ├── node_01_generate_data.py      # Generate synthetic data
│   │   ├── node_02_preprocess_data.py    # Data preprocessing
│   │   ├── node_03_feature_engineering.py# Advanced feature creation
│   │   ├── node_04_train_test_split.py   # Train/test data splitting
│   │   ├── node_05_train_model.py        # XGBoost model training
│   │   └── node_06_evaluate_model.py     # Model evaluation & SHAP
│   └── utils/                   # Shared utilities
│       ├── data_catalog.py      # Data catalog system
│       ├── pipeline.py          # Central pipeline manifest
│       ├── run_node.py          # Container entrypoint
│       └── snowflake_utils.py   # Snowflake connection utilities
├── data/
│   └── sample_customer_data.py  # Sample data generation
├── jobs/                        # Snowflake ML Job YAML specifications
│   ├── job-generate-data.yaml
│   ├── job-preprocess-data.yaml
│   ├── job-feature-engineering.yaml
│   ├── job-train-test-split.yaml
│   ├── job-train-model.yaml
│   └── job-evaluate-model.yaml
├── tasks/
│   └── create_pipeline_tasks.sql # SQL to create Snowflake Task DAG
├── .github/workflows/
│   └── ci-cd.yml                # GitHub Actions CI/CD pipeline
├── main.py                      # CLI orchestrator for pipeline management
├── test_pipeline.py            # End-to-end pipeline testing
├── config.env                  # Environment variables template
├── USAGE.md                    # Detailed usage guide
├── Dockerfile                  # Container definition
├── environment.yml            # Conda environment specification
└── requirements.txt           # Python dependencies (pip fallback)
```

## 🚀 Quick Start

Get up and running with the MLOps framework in 4 simple steps:

```bash
# 1. Setup environment
cp config.env .env
# Edit .env with your Snowflake credentials

# 2. Create conda environment and install dependencies
conda env create -f environment.yml
conda activate mlops-framework

# 3. Test locally
python main.py list                    # See available nodes
python main.py run                     # Run full pipeline locally
python test_pipeline.py               # Comprehensive testing

# 4. Deploy to Snowflake
python main.py tasks create           # Create Snowflake tasks
python main.py tasks resume           # Start pipeline execution
python main.py tasks status           # Monitor progress
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## 📋 Prerequisites

1. **Python Environment**:
   - **Conda** (recommended) - [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - **OR Python 3.10+** with pip

2. **Snowflake Account** with:
   - Snowpark Container Services enabled
   - Compute Pools available
   - Feature Store and Model Registry access

3. **Required Privileges**:
   - CREATE COMPUTE POOL
   - CREATE SERVICE
   - CREATE TASK
   - CREATE ROLE
   - Feature Store and Model Registry permissions

## 🔧 Manual Setup

For advanced users who want to set up the framework manually:

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sf_ml_framework
   ```

2. **Configure environment**:
   ```bash
   # Copy the environment template and edit with your credentials
   cp config.env .env
   # Edit .env file with your Snowflake account details
   ```

3. **Install dependencies**:
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate mlops-framework
   
   # OR using pip (alternative)
   pip install -r requirements.txt
   ```

### Deployment

1. **Create Snowflake infrastructure**:
   ```sql
   -- Run the SQL script to create compute pools, roles, and tasks
   -- Update placeholders with your actual values first
   @tasks/create_pipeline_tasks.sql
   ```

2. **Build and push Docker image**:
   ```bash
   # Build the image
   docker build -t mlops_framework:latest .
   
   # Tag for Snowflake registry
   docker tag mlops_framework:latest <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
   
   # Push to Snowflake registry
   docker push <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
   ```

3. **Update job specifications**:
   - Replace `<SNOWFLAKE_REGISTRY_URL>` in all job YAML files
   - Replace `<SNOWFLAKE_ACCOUNT>`, `<SNOWFLAKE_DATABASE>`, `<SNOWFLAKE_SCHEMA>` placeholders

4. **Start the pipeline**:
   ```sql
   -- Resume all tasks to activate the pipeline
   ALTER TASK task_generate_sample_data RESUME;
   ALTER TASK task_preprocess_data RESUME;
   -- ... etc for all tasks
   ```

## 🏠 Execution Modes

The framework automatically detects and supports two execution modes:

### Local Mode
- **Purpose**: Development, testing, and debugging
- **Data Storage**: Files saved to `data/` folder as CSV files
- **Model Storage**: Models saved as pickle files locally
- **Feature Store**: None (uses local files)
- **Model Registry**: None (uses local files)
- **Detection**: Automatic when `feature_store=None`

### Deployment Mode  
- **Purpose**: Production execution on Snowflake
- **Data Storage**: Snowflake Feature Store (FeatureViews)
- **Model Storage**: Snowflake Model Registry
- **Feature Store**: Real Snowflake Feature Store instance
- **Model Registry**: Real Snowflake Model Registry instance
- **Detection**: Automatic when Snowflake services are provided

### Mode Detection in Nodes

Each node automatically detects the execution mode:

```python
def node_function(feature_store, model_registry, inputs, outputs):
    # Detect execution mode
    is_local_mode = feature_store is None
    
    if is_local_mode:
        # LOCAL MODE: Save to data/ folder
        df.to_csv("data/output.csv")
        logger.info("💾 Saving to local data/ folder...")
    else:
        # DEPLOYMENT MODE: Save to Snowflake Feature Store
        feature_store.register_feature_view(feature_view)
        logger.info("☁️  Saving to Snowflake Feature Store...")
```

### Testing Both Modes

```bash
# Test local vs deployment mode functionality
python test_local_deployment_modes.py
```

## 🔧 Local Development

### Testing Individual Nodes

```bash
# Test data generation node
cd src/nodes
python node_01_generate_data.py

# Test preprocessing node  
python node_02_preprocess_data.py
```

### Running the Pipeline Runner

```bash
# Test the node runner
python src/utils/run_node.py --node generate_sample_data --inputs '{"num_customers": 50}'
```

### Sample Data Generation

```bash
# Generate sample customer data
python data/sample_customer_data.py
```

## 📊 Pipeline Overview

The framework includes a complete customer churn prediction pipeline with 6 nodes:

### 1. **Generate Sample Data** (`node_01_generate_data.py`)
- Creates synthetic customer data (100000 customers, 9 features)
- Realistic churn rate (~12%)
- Stores in Snowflake Feature Store

### 2. **Preprocess Data** (`node_02_preprocess_data.py`)
- Missing value imputation
- Outlier handling and capping
- Data validation
- Feature flags creation (high value, new customer, low engagement)

### 3. **Feature Engineering** (`node_03_feature_engineering.py`)
- Advanced feature creation (40+ new features)
- Derived metrics (revenue per product, session intensity)
- Categorical encoding (one-hot, label encoding)
- Interaction features and risk scoring

### 4. **Train/Test Split** (`node_04_train_test_split.py`)
- Stratified sampling maintaining target distribution
- Data quality validation
- Split quality verification

### 5. **Model Training** (`node_05_train_model.py`)
- XGBoost classifier with cross-validation
- Hyperparameter optimization
- Feature importance analysis
- Model artifact management

### 6. **Model Evaluation** (`node_06_evaluate_model.py`)
- Performance metrics (AUC, accuracy, precision/recall)
- SHAP explanations for interpretability
- Visualization generation (ROC curves, confusion matrix)
- Model registry storage

## ⚙️ Configuration

### Node Configuration

Each node can specify its own resources and configuration:

```python
NodeConfig(
    name="train_model",
    function=train_model_function,
    compute_pool_size="LARGE",
    memory_request="8Gi",
    cpu_request="4",
    timeout_minutes=120,
    dependencies=["feature_engineering"]
)
```

### Feature Versioning

Features support user-defined versioning:

```python
inputs = {
    "source_feature_view": "customer_features",
    "version": "v1.2.0"  # User-defined version string
}
```

## 🔐 Security

### Authentication
- **Recommended**: Key-pair authentication
- **Fallback**: Password-based authentication
- **Secrets**: Stored in Snowflake Secrets or environment variables

### Access Control
- Custom `MLOPS_ROLE` role with minimal required privileges
- Database-per-environment separation
- Network access controls

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### End-to-End Tests
```bash
python scripts/run_e2e_test.py
```

## 🚀 CI/CD

The framework includes GitHub Actions workflows for:

- **Unit Testing**: Runs on every commit
- **Integration Testing**: Runs against Snowflake test environment
- **Security Scanning**: Trivy vulnerability scanning
- **Multi-Environment Deployment**: 
  - `develop` → test environment
  - `main` → production environment

## 📈 Monitoring

### Task Monitoring
```sql
-- View task history
SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) 
ORDER BY SCHEDULED_TIME DESC LIMIT 10;

-- Check task status
SHOW TASKS LIKE 'task_%';
```

### Compute Pool Monitoring
```sql
-- Monitor compute pool usage
SELECT * FROM TABLE(INFORMATION_SCHEMA.COMPUTE_POOL_HISTORY());
```

## 🔄 Operational Procedures

### Manual Node Execution
```sql
-- Execute a specific node manually
EXECUTE SERVICE job-generate-data 
IN COMPUTE POOL MLOPS_COMPUTE_POOL;
```

### Pipeline Recovery
```sql
-- Restart from a specific node after fixing issues
ALTER TASK task_preprocess_data RESUME;
```

### Scaling Compute Pools
```sql
-- Scale up for heavy workloads
ALTER COMPUTE POOL MLOPS_COMPUTE_POOL SET MAX_NODES = 20;
```

## 📚 Advanced Usage

### Custom Nodes

1. Create node function in `src/nodes/`
2. Register in `src/utils/pipeline.py`
3. Create job YAML in `jobs/`
4. Add task to `tasks/create_pipeline_tasks.sql`

### Multi-Environment Deployment

The framework supports multiple environments through:
- Database-per-environment
- Environment-specific secrets
- Branch-based deployment strategy
