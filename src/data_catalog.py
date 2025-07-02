"""
Data Catalog Utility

Provides functions to read the data catalog and resolve logical data names 
to their physical locations and metadata.
"""

import os
import yaml
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import glob

class DataCatalog:
    """
    Data catalog for managing dataset and model locations across local and deployment environments.
    """
    
    def __init__(self, catalog_path: str = "conf/data_catalog.yaml"):
        """
        Initialize the data catalog.
        
        Args:
            catalog_path: Path to the catalog YAML file
        """
        self.catalog_path = catalog_path
        self._catalog: Dict[str, Any] = {}
        self._load_catalog()
    
    def _load_catalog(self):
        """Load the catalog from YAML file."""
        try:
            with open(self.catalog_path, 'r') as f:
                self._catalog = yaml.safe_load(f)
                if self._catalog is None:
                    self._catalog = {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Data catalog not found at: {self.catalog_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in data catalog: {e}")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get complete information about a dataset.
        
        Args:
            dataset_name: Logical name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        if dataset_name not in self._catalog.get('datasets', {}):
            raise ValueError(f"Dataset '{dataset_name}' not found in catalog")
        
        return self._catalog['datasets'][dataset_name]
    
    def get_default_version(self, dataset_name: str, item_type: str = 'datasets') -> str:
        """
        Get the default version for a dataset or model from the catalog.
        
        Args:
            dataset_name: Logical name of the dataset/model
            item_type: Type of item ('datasets', 'models', 'evaluations')
            
        Returns:
            Default version string
        """
        items = self._catalog.get(item_type, {})
        if dataset_name not in items:
            raise ValueError(f"{item_type.title()[:-1]} '{dataset_name}' not found in catalog")
        
        item_info = items[dataset_name]
        return item_info.get('version', self._catalog.get('settings', {}).get('default_version', 'v1'))
    
    def resolve_version(self, dataset_name: str, version: Optional[str] = None, 
                       item_type: str = 'datasets') -> str:
        """
        Resolve version string to actual version to use.
        
        Args:
            dataset_name: Logical name of the dataset/model
            version: Requested version (None, 'latest', 'current', or specific version)
            item_type: Type of item ('datasets', 'models', 'evaluations')
            
        Returns:
            Resolved version string
        """
        if version is None or version == 'current':
            # Use default version from catalog
            return self.get_default_version(dataset_name, item_type)
        elif version == 'latest':
            # Find latest version available on disk
            return self._find_latest_version(dataset_name, item_type)
        else:
            # Use specified version
            return version
    
    def _find_latest_version(self, dataset_name: str, item_type: str = 'datasets') -> str:
        """
        Find the latest version available on disk by scanning files.
        
        Args:
            dataset_name: Logical name of the dataset/model
            item_type: Type of item ('datasets', 'models', 'evaluations')
            
        Returns:
            Latest version string found
        """
        items = self._catalog.get(item_type, {})
        if dataset_name not in items:
            raise ValueError(f"{item_type.title()[:-1]} '{dataset_name}' not found in catalog")
        
        item_info = items[dataset_name]
        local_info = item_info.get('local', {})
        directory = local_info.get('directory', 'data')
        
        if item_type == 'datasets':
            filename_pattern = local_info.get('filename_pattern', '{name}.csv')
        elif item_type == 'models':
            filename_pattern = local_info.get('filename_pattern', 'model_{name}_{version}.pkl')
        else:
            filename_pattern = local_info.get('filename_pattern', '{name}_{version}.json')
        
        # Convert pattern to glob pattern and regex for version extraction
        glob_pattern = filename_pattern.format(name=dataset_name, version='*')
        glob_path = os.path.join(directory, glob_pattern)
        
        # Find all matching files
        matching_files = glob.glob(glob_path)
        
        if not matching_files:
            # No files found, return default version
            return self.get_default_version(dataset_name, item_type)
        
        # Extract versions from filenames using improved regex
        versions = []
        
        # Create regex pattern from filename pattern - more robust approach
        # Replace placeholders with actual values and version with capture group
        escaped_name = re.escape(dataset_name)
        
        # Build regex pattern step by step
        regex_pattern = filename_pattern
        regex_pattern = regex_pattern.replace('{name}', escaped_name)
        regex_pattern = regex_pattern.replace('{version}', r'(v\d+)')
        
        # Escape special regex characters that might be in the pattern
        regex_pattern = regex_pattern.replace('.', r'\.')
        
        for filepath in matching_files:
            filename = os.path.basename(filepath)
            match = re.search(regex_pattern, filename)
            if match and match.groups():
                version = match.group(1)
                versions.append(version)
        
        if not versions:
            # No versions found in filenames, return default
            return self.get_default_version(dataset_name, item_type)
        
        # Sort versions numerically (extract numeric part and sort)
        def version_key(v):
            # Extract numeric part from version string (e.g., 'v10' -> 10)
            try:
                return int(v[1:]) if v.startswith('v') and v[1:].isdigit() else 0
            except (ValueError, IndexError):
                return 0
        
        versions.sort(key=version_key, reverse=True)
        latest_version = versions[0]
        
        return latest_version
    
    def get_local_path(self, dataset_name: str, name_override: Optional[str] = None, 
                      version: Optional[str] = None, create_dir: bool = False) -> str:
        """
        Get the local file path for a dataset.
        
        Args:
            dataset_name: Logical name of the dataset
            name_override: Override the name used in the filename pattern
            version: Version string (None='current', 'latest'=find latest, or specific version)
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Complete file path
        """
        dataset_info = self.get_dataset_info(dataset_name)
        local_info = dataset_info.get('local', {})
        
        directory = local_info.get('directory', 'data')
        filename_pattern = local_info.get('filename_pattern', '{name}.csv')
        
        # Resolve version
        resolved_version = self.resolve_version(dataset_name, version, 'datasets')
        
        # Use override name or dataset name
        name = name_override or dataset_name
        
        # Format the filename
        filename = filename_pattern.format(name=name, version=resolved_version)
        
        # Create full path
        filepath = os.path.join(directory, filename)
        
        # Create directory if requested
        if create_dir:
            os.makedirs(directory, exist_ok=True)
        
        return filepath
    
    def get_local_metadata_path(self, dataset_name: str, name_override: Optional[str] = None,
                               version: Optional[str] = None, create_dir: bool = False) -> str:
        """
        Get the local metadata file path for a dataset.
        
        Args:
            dataset_name: Logical name of the dataset
            name_override: Override the name used in the filename pattern
            version: Version string (None='current', 'latest'=find latest, or specific version)
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Complete metadata file path
        """
        dataset_info = self.get_dataset_info(dataset_name)
        local_info = dataset_info.get('local', {})
        
        directory = local_info.get('directory', 'data')
        metadata_pattern = local_info.get('metadata_pattern', 'metadata_{name}_{version}.json')
        
        # Resolve version
        resolved_version = self.resolve_version(dataset_name, version, 'datasets')
        
        # Use override name or dataset name
        name = name_override or dataset_name
        
        # Format the filename
        filename = metadata_pattern.format(name=name, version=resolved_version)
        
        # Create full path
        filepath = os.path.join(directory, filename)
        
        # Create directory if requested
        if create_dir:
            os.makedirs(directory, exist_ok=True)
        
        return filepath
    
    def get_snowflake_feature_view(self, dataset_name: str) -> str:
        """
        Get the Snowflake feature view name for a dataset.
        
        Args:
            dataset_name: Logical name of the dataset
            
        Returns:
            Snowflake feature view name
        """
        dataset_info = self.get_dataset_info(dataset_name)
        snowflake_info = dataset_info.get('snowflake', {})
        
        return snowflake_info.get('feature_view', dataset_name)
    
    def find_local_file(self, dataset_name: str, name_override: Optional[str] = None,
                       version: Optional[str] = None) -> Optional[str]:
        """
        Find existing local file for a dataset, trying multiple possible names.
        
        Args:
            dataset_name: Logical name of the dataset
            name_override: Override the name used in the filename pattern
            version: Version string (None='current', 'latest'=find latest, or specific version)
            
        Returns:
            Path to existing file or None if not found
        """
        dataset_info = self.get_dataset_info(dataset_name)
        local_info = dataset_info.get('local', {})
        
        directory = local_info.get('directory', 'data')
        filename_pattern = local_info.get('filename_pattern', '{name}.csv')
        
        # Resolve version
        resolved_version = self.resolve_version(dataset_name, version, 'datasets')
        
        # Try multiple possible names
        name_candidates = []
        if name_override:
            name_candidates.append(name_override)
        name_candidates.extend([dataset_name, dataset_name.replace('_', '-')])
        
        for name in name_candidates:
            filename = filename_pattern.format(name=name, version=resolved_version)
            filepath = os.path.join(directory, filename)
            
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete information about a model.
        
        Args:
            model_name: Logical name of the model
            
        Returns:
            Dictionary containing model information
        """
        if model_name not in self._catalog.get('models', {}):
            raise ValueError(f"Model '{model_name}' not found in catalog")
        
        return self._catalog['models'][model_name]
    
    def get_model_local_path(self, model_name: str, version: Optional[str] = None, 
                            create_dir: bool = False) -> str:
        """
        Get the local file path for a model.
        
        Args:
            model_name: Logical name of the model
            version: Version string (None='current', 'latest'=find latest, or specific version)
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Complete model file path
        """
        model_info = self.get_model_info(model_name)
        local_info = model_info.get('local', {})
        
        directory = local_info.get('directory', 'data')
        filename_pattern = local_info.get('filename_pattern', 'model_{name}_{version}.pkl')
        
        # Resolve version
        resolved_version = self.resolve_version(model_name, version, 'models')
        
        # Format the filename
        filename = filename_pattern.format(name=model_name, version=resolved_version)
        
        # Create full path
        filepath = os.path.join(directory, filename)
        
        # Create directory if requested
        if create_dir:
            os.makedirs(directory, exist_ok=True)
        
        return filepath
    
    def get_model_metadata_path(self, model_name: str, version: Optional[str] = None,
                               create_dir: bool = False) -> str:
        """
        Get the local metadata file path for a model.
        
        Args:
            model_name: Logical name of the model
            version: Version string (None='current', 'latest'=find latest, or specific version)
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Complete model metadata file path
        """
        model_info = self.get_model_info(model_name)
        local_info = model_info.get('local', {})
        
        directory = local_info.get('directory', 'data')
        metadata_pattern = local_info.get('metadata_pattern', 'model_metadata_{name}_{version}.json')
        
        # Resolve version
        resolved_version = self.resolve_version(model_name, version, 'models')
        
        # Format the filename
        filename = metadata_pattern.format(name=model_name, version=resolved_version)
        
        # Create full path
        filepath = os.path.join(directory, filename)
        
        # Create directory if requested
        if create_dir:
            os.makedirs(directory, exist_ok=True)
        
        return filepath
    
    def get_evaluation_plot_path(self, plot_type: str, model_name: str, model_version: Optional[str] = None) -> str:
        """
        Get the path for an evaluation plot.
        
        Args:
            plot_type: Type of plot (confusion_matrix, roc_curve, etc.)
            model_name: Name of the model
            model_version: Version of the model (None='current', 'latest'=find latest, or specific version)
            
        Returns:
            Complete plot file path
        """
        evaluation_info = self._catalog.get('evaluations', {}).get('model_evaluation', {})
        local_info = evaluation_info.get('local', {})
        plots_config = local_info.get('plots', {})
        
        if plot_type not in plots_config:
            raise ValueError(f"Plot type '{plot_type}' not configured in catalog")
        
        directory = local_info.get('directory', 'data')
        plot_pattern = plots_config[plot_type]
        
        # Resolve version for the evaluation (not the model itself)
        resolved_version = self.resolve_version('model_evaluation', model_version, 'evaluations')
        
        # Format the filename
        filename = plot_pattern.format(model_name=model_name, model_version=resolved_version)
        
        # Create full path
        return os.path.join(directory, filename)
    
    def list_datasets(self) -> List[str]:
        """
        Get list of all available datasets.
        
        Returns:
            List of dataset names
        """
        return list(self._catalog.get('datasets', {}).keys())
    
    def list_models(self) -> List[str]:
        """
        Get list of all available models.
        
        Returns:
            List of model names
        """
        return list(self._catalog.get('models', {}).keys())


# Global instance for easy access
_catalog_instance = None

def get_catalog() -> DataCatalog:
    """
    Get the global data catalog instance.
    
    Returns:
        DataCatalog instance
    """
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = DataCatalog()
    return _catalog_instance

def get_dataset_path(dataset_name: str, name_override: Optional[str] = None, 
                    version: Optional[str] = None, create_dir: bool = False) -> str:
    """
    Convenience function to get local dataset path.
    
    Args:
        dataset_name: Logical name of the dataset
        name_override: Override the name used in the filename pattern
        version: Version string (None='current', 'latest'=find latest, or specific version)
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Complete file path
    """
    return get_catalog().get_local_path(dataset_name, name_override, version, create_dir)

def get_feature_view_name(dataset_name: str) -> str:
    """
    Convenience function to get Snowflake feature view name.
    
    Args:
        dataset_name: Logical name of the dataset
        
    Returns:
        Snowflake feature view name
    """
    return get_catalog().get_snowflake_feature_view(dataset_name)

def find_dataset_file(dataset_name: str, name_override: Optional[str] = None,
                     version: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to find existing dataset file.
    
    Args:
        dataset_name: Logical name of the dataset
        name_override: Override the name used in the filename pattern
        version: Version string (None='current', 'latest'=find latest, or specific version)
        
    Returns:
        Path to existing file or None if not found
    """
    return get_catalog().find_local_file(dataset_name, name_override, version) 