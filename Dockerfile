# Use official miniconda image for better dependency management
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first (for better Docker layer caching)
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "mlops-framework", "/bin/bash", "-c"]

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Copy configuration files
COPY requirements.txt .
COPY config.env .

# Set the conda environment in PATH
ENV PATH /opt/conda/envs/mlops-framework/bin:$PATH

# Activate the environment and set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlops-framework", "python", "/app/src/run_node.py"]

# Default command (can be overridden)
CMD ["--help"] 