-- Create the role for MLOps framework
CREATE ROLE IF NOT EXISTS MLOPS_ROLE
  COMMENT = 'Role for MLOps framework operations';

-- Grant the role to your user
GRANT ROLE MLOPS_ROLE TO USER JARCHEN;

-- Create a dedicated database and warehouse
CREATE OR REPLACE DATABASE MLOPS_DATABASE;
CREATE OR REPLACE SCHEMA MLOPS_SCHEMA;

-- Create the task warehouse
CREATE WAREHOUSE IF NOT EXISTS MLOPS_TASK_WH
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = FALSE
  COMMENT = 'Warehouse for MLOps framework task orchestration';

-- Grant warehouse privileges
GRANT USAGE ON WAREHOUSE MLOPS_TASK_WH TO ROLE MLOPS_ROLE;

-- Grant database and schema privileges (adjust database and schema names as needed)
GRANT USAGE ON DATABASE MLOPS_DATABASE TO ROLE MLOPS_ROLE;
GRANT USAGE ON SCHEMA MLOPS_DATABASE.MLOPS_SCHEMA TO ROLE MLOPS_ROLE;
GRANT CREATE TABLE, CREATE VIEW, CREATE STAGE ON SCHEMA MLOPS_DATABASE.MLOPS_SCHEMA TO ROLE MLOPS_ROLE;

GRANT CREATE COMPUTE POOL ON ACCOUNT TO ROLE MLOPS_ROLE;

GRANT CREATE SERVICE ON SCHEMA MLOPS_DATABASE.MLOPS_SCHEMA TO ROLE MLOPS_ROLE;