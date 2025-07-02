-- MLOps Framework Pipeline Tasks
-- This script creates the Snowflake Task DAG for the customer churn prediction pipeline

-- First, create the compute pool for container services
CREATE COMPUTE POOL IF NOT EXISTS MLOPS_COMPUTE_POOL
  MIN_NODES = 1
  MAX_NODES = 10
  INSTANCE_FAMILY = CPU_X64_XS
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = FALSE
  COMMENT = 'Compute pool for MLOps framework container services';

-- Create the task warehouse
CREATE WAREHOUSE IF NOT EXISTS MLOPS_TASK_WH
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = FALSE
  COMMENT = 'Warehouse for MLOps framework task orchestration';

-- Create the role for MLOps framework
CREATE ROLE IF NOT EXISTS MLOPS_ROLE
  COMMENT = 'Role for MLOps framework operations';

-- Grant necessary privileges to the role
GRANT USAGE ON WAREHOUSE MLOPS_TASK_WH TO ROLE MLOPS_ROLE;
GRANT USAGE, MONITOR ON COMPUTE POOL MLOPS_COMPUTE_POOL TO ROLE MLOPS_ROLE;
GRANT CREATE SERVICE ON COMPUTE POOL MLOPS_COMPUTE_POOL TO ROLE MLOPS_ROLE;

-- Grant database and schema privileges (adjust database and schema names as needed)
GRANT USAGE ON DATABASE <SNOWFLAKE_DATABASE> TO ROLE MLOPS_ROLE;
GRANT USAGE ON SCHEMA <SNOWFLAKE_DATABASE>.<SNOWFLAKE_SCHEMA> TO ROLE MLOPS_ROLE;
GRANT CREATE TABLE, CREATE VIEW, CREATE STAGE ON SCHEMA <SNOWFLAKE_DATABASE>.<SNOWFLAKE_SCHEMA> TO ROLE MLOPS_ROLE;

-- Task 1: Generate Sample Data
CREATE OR REPLACE TASK task_generate_sample_data
  WAREHOUSE = MLOPS_TASK_WH
  SCHEDULE = 'USING CRON 0 6 * * * UTC'  -- Daily at 6 AM UTC
  USER_TASK_TIMEOUT_MS = 3600000  -- 1 hour timeout
  COMMENT = 'Generate synthetic customer data for churn prediction'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-generate-data
spec:
  containers:
    - name: generate-data-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "generate_sample_data"
        - "--inputs"
        - '{"num_customers": 100000, "random_seed": 42}'
        - "--outputs"
        - '{"table_name": "RAW_CUSTOMER_DATA", "feature_view_name": "raw_customer_features"}'
      resources:
        requests:
          memory: "1Gi"
          cpu: "0.5"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Task 2: Preprocess Data
CREATE OR REPLACE TASK task_preprocess_data
  WAREHOUSE = MLOPS_TASK_WH
  AFTER task_generate_sample_data
  USER_TASK_TIMEOUT_MS = 3600000  -- 1 hour timeout
  COMMENT = 'Clean and preprocess raw customer data'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-preprocess-data
spec:
  containers:
    - name: preprocess-data-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "preprocess_data"
        - "--inputs"
        - '{"source_feature_view": "raw_customer_features", "version": "v1"}'
        - "--outputs"
        - '{"feature_view_name": "preprocessed_customer_features", "version": "v1"}'
      resources:
        requests:
          memory: "2Gi"
          cpu: "1.0"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Task 3: Feature Engineering
CREATE OR REPLACE TASK task_engineer_features
  WAREHOUSE = MLOPS_TASK_WH
  AFTER task_preprocess_data
  USER_TASK_TIMEOUT_MS = 3600000  -- 1 hour timeout
  COMMENT = 'Create engineered features for model training'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-engineer-features
spec:
  containers:
    - name: engineer-features-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "engineer_features"
        - "--inputs"
        - '{"source_feature_view": "preprocessed_customer_features", "version": "v1"}'
        - "--outputs"
        - '{"feature_view_name": "engineered_customer_features", "version": "v1"}'
      resources:
        requests:
          memory: "4Gi"
          cpu: "2.0"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Task 4: Train/Test Split
CREATE OR REPLACE TASK task_train_test_split
  WAREHOUSE = MLOPS_TASK_WH
  AFTER task_engineer_features
  USER_TASK_TIMEOUT_MS = 3600000  -- 1 hour timeout
  COMMENT = 'Split data into training and testing sets'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-train-test-split
spec:
  containers:
    - name: train-test-split-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "create_train_test_split"
        - "--inputs"
        - '{"source_feature_view": "engineered_customer_features", "version": "v1", "test_size": 0.2, "random_seed": 42}'
        - "--outputs"
        - '{"train_feature_view": "train_customer_features", "test_feature_view": "test_customer_features", "version": "v1"}'
      resources:
        requests:
          memory: "2Gi"
          cpu: "1.0"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Task 5: Train Model
CREATE OR REPLACE TASK task_train_model
  WAREHOUSE = MLOPS_TASK_WH
  AFTER task_train_test_split
  USER_TASK_TIMEOUT_MS = 7200000  -- 2 hour timeout
  COMMENT = 'Train XGBoost model for churn prediction'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-train-model
spec:
  containers:
    - name: train-model-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "train_model"
        - "--inputs"
        - '{"train_feature_view": "train_customer_features", "version": "v1", "target_column": "churned", "model_params": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "random_state": 42}}'
        - "--outputs"
        - '{"model_name": "customer_churn_xgboost", "model_version": "v1"}'
      resources:
        requests:
          memory: "8Gi"
          cpu: "4.0"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Task 6: Evaluate Model
CREATE OR REPLACE TASK task_evaluate_model
  WAREHOUSE = MLOPS_TASK_WH
  AFTER task_train_model
  USER_TASK_TIMEOUT_MS = 5400000  -- 1.5 hour timeout
  COMMENT = 'Evaluate model performance and generate SHAP plots'
AS
  EXECUTE SERVICE
  IN COMPUTE POOL MLOPS_COMPUTE_POOL
  FROM SPECIFICATION $$
apiVersion: v1
kind: Service
metadata:
  name: job-evaluate-model
spec:
  containers:
    - name: evaluate-model-container
      image: <SNOWFLAKE_REGISTRY_URL>/mlops_framework:latest
      args: 
        - "--node"
        - "evaluate_model"
        - "--inputs"
        - '{"model_name": "customer_churn_xgboost", "model_version": "v1", "test_feature_view": "test_customer_features", "version": "v1", "target_column": "churned"}'
        - "--outputs"
        - '{"evaluation_table": "MODEL_EVALUATION_RESULTS", "shap_plots_stage": "SHAP_PLOTS"}'
      resources:
        requests:
          memory: "4Gi"
          cpu: "2.0"
      env:
        - name: SNOWFLAKE_USER
          value: "<SNOWFLAKE_USER>"
        - name: SNOWFLAKE_ACCOUNT
          value: "<SNOWFLAKE_ACCOUNT>"
        - name: SNOWFLAKE_WAREHOUSE
          value: "MLOPS_TASK_WH"
        - name: SNOWFLAKE_DATABASE
          value: "<SNOWFLAKE_DATABASE>"
        - name: SNOWFLAKE_SCHEMA
          value: "<SNOWFLAKE_SCHEMA>"
        - name: SNOWFLAKE_ROLE
          value: "MLOPS_ROLE"
  min_instances: 1
  max_instances: 1
$$;

-- Enable email notifications for task failures (optional)
-- CREATE NOTIFICATION INTEGRATION IF NOT EXISTS email_integration
--   TYPE = EMAIL
--   ENABLED = TRUE
--   ALLOWED_RECIPIENTS = ('admin@company.com', 'mlops-team@company.com');

-- Resume tasks to activate the pipeline
ALTER TASK task_generate_sample_data RESUME;
ALTER TASK task_preprocess_data RESUME;
ALTER TASK task_engineer_features RESUME;
ALTER TASK task_train_test_split RESUME;
ALTER TASK task_train_model RESUME;
ALTER TASK task_evaluate_model RESUME;

-- Show task status
SHOW TASKS LIKE 'task_%';

-- Show task history
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY()) ORDER BY SCHEDULED_TIME DESC LIMIT 10; 