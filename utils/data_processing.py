"""
Quantum Edge AI Platform - Data Processing Utilities

Advanced data processing, transformation, and analysis utilities
optimized for edge AI and quantum computing workloads.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib
from enum import Enum
import json
from pathlib import Path
import pickle
import gzip
import threading
import queue
import concurrent.futures

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats"""
    NUMPY = "numpy"
    PANDAS = "pandas"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    HDF5 = "hdf5"

class DataType(Enum):
    """Data type classifications"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    TABULAR = "tabular"

@dataclass
class DataProfile:
    """Data profiling information"""
    shape: Tuple[int, ...]
    data_type: DataType
    dtype: str
    memory_usage: int
    null_count: int
    unique_count: int = 0
    statistics: Dict[str, float] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DataTransformation:
    """Data transformation specification"""
    name: str
    transformation_type: str
    parameters: Dict[str, Any]
    input_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    applied_at: datetime = field(default_factory=datetime.utcnow)

class DataProcessor:
    """Advanced data processor for edge AI"""

    def __init__(self, chunk_size: int = 1000, max_workers: int = 4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.transformations = []
        self.data_cache = {}
        self.processing_queue = queue.Queue()

    def load_data(self, source: Union[str, Path, np.ndarray, pd.DataFrame],
                 format: DataFormat = None, **kwargs) -> Any:
        """Load data from various sources"""
        if isinstance(source, np.ndarray):
            return source
        elif isinstance(source, pd.DataFrame):
            return source
        elif isinstance(source, (str, Path)):
            path = Path(source)

            if format is None:
                # Auto-detect format from extension
                format = self._detect_format(path)

            if format == DataFormat.NUMPY:
                return np.load(path, **kwargs)
            elif format == DataFormat.PANDAS:
                return pd.read_pickle(path, **kwargs)
            elif format == DataFormat.JSON:
                with open(path, 'r') as f:
                    return json.load(f)
            elif format == DataFormat.CSV:
                return pd.read_csv(path, **kwargs)
            elif format == DataFormat.PARQUET:
                return pd.read_parquet(path, **kwargs)
            elif format == DataFormat.PICKLE:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            elif format == DataFormat.HDF5:
                return pd.read_hdf(path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")

        raise ValueError(f"Unsupported source type: {type(source)}")

    def _detect_format(self, path: Path) -> DataFormat:
        """Detect data format from file extension"""
        suffix = path.suffix.lower()

        format_map = {
            '.npy': DataFormat.NUMPY,
            '.npz': DataFormat.NUMPY,
            '.pkl': DataFormat.PICKLE,
            '.json': DataFormat.JSON,
            '.csv': DataFormat.CSV,
            '.parquet': DataFormat.PARQUET,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5
        }

        return format_map.get(suffix, DataFormat.PICKLE)

    def save_data(self, data: Any, destination: Union[str, Path],
                 format: DataFormat = None, compress: bool = False, **kwargs):
        """Save data to various formats"""
        path = Path(destination)

        if format is None:
            format = self._detect_format(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if compress and format in [DataFormat.JSON, DataFormat.CSV]:
            # Use gzip compression
            with gzip.open(path, 'wt' if format == DataFormat.JSON else 'wb') as f:
                if format == DataFormat.JSON:
                    json.dump(data, f, **kwargs)
                else:
                    data.to_csv(f, **kwargs)
            return

        if format == DataFormat.NUMPY:
            np.save(path, data, **kwargs)
        elif format == DataFormat.PANDAS:
            data.to_pickle(path, **kwargs)
        elif format == DataFormat.JSON:
            with open(path, 'w') as f:
                json.dump(data, f, **kwargs)
        elif format == DataFormat.CSV:
            data.to_csv(path, **kwargs)
        elif format == DataFormat.PARQUET:
            data.to_parquet(path, **kwargs)
        elif format == DataFormat.PICKLE:
            with open(path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        elif format == DataFormat.HDF5:
            data.to_hdf(path, key='data', **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def profile_data(self, data: Any) -> DataProfile:
        """Generate comprehensive data profile"""
        if isinstance(data, np.ndarray):
            return self._profile_numpy_array(data)
        elif isinstance(data, pd.DataFrame):
            return self._profile_pandas_dataframe(data)
        elif isinstance(data, list):
            return self._profile_list(data)
        else:
            return self._profile_generic(data)

    def _profile_numpy_array(self, array: np.ndarray) -> DataProfile:
        """Profile NumPy array"""
        statistics = {}
        if array.size > 0:
            statistics = {
                'mean': float(np.mean(array)),
                'std': float(np.std(array)),
                'min': float(np.min(array)),
                'max': float(np.max(array)),
                'median': float(np.median(array))
            }

        return DataProfile(
            shape=array.shape,
            data_type=DataType.NUMERIC,
            dtype=str(array.dtype),
            memory_usage=array.nbytes,
            null_count=int(np.isnan(array).sum() if np.issubdtype(array.dtype, np.floating) else 0),
            unique_count=len(np.unique(array)),
            statistics=statistics,
            sample_values=array.flatten()[:10].tolist()
        )

    def _profile_pandas_dataframe(self, df: pd.DataFrame) -> DataProfile:
        """Profile pandas DataFrame"""
        memory_usage = df.memory_usage(deep=True).sum()
        null_count = df.isnull().sum().sum()

        statistics = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            statistics = df[numeric_cols].describe().to_dict()

        return DataProfile(
            shape=df.shape,
            data_type=DataType.TABULAR,
            dtype="DataFrame",
            memory_usage=int(memory_usage),
            null_count=int(null_count),
            unique_count=sum(df.nunique()),
            statistics=statistics,
            sample_values=df.head(5).to_dict('records')
        )

    def _profile_list(self, data: list) -> DataProfile:
        """Profile Python list"""
        return DataProfile(
            shape=(len(data),),
            data_type=DataType.TABULAR,
            dtype=f"list[{type(data[0]).__name__ if data else 'empty'}]",
            memory_usage=len(data) * 28,  # Rough estimate
            null_count=sum(1 for x in data if x is None),
            unique_count=len(set(data)),
            sample_values=data[:10]
        )

    def _profile_generic(self, data: Any) -> DataProfile:
        """Profile generic data"""
        return DataProfile(
            shape=(),
            data_type=DataType.TABULAR,
            dtype=type(data).__name__,
            memory_usage=0,
            null_count=0,
            sample_values=[data] if not isinstance(data, (list, dict)) else []
        )

    def normalize_data(self, data: np.ndarray, method: str = 'standard',
                      axis: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize data using various methods"""
        if method == 'standard':
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            normalized = (data - mean) / std

            params = {'method': 'standard', 'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val

            params = {'method': 'minmax', 'min': min_val, 'max': max_val}

        elif method == 'robust':
            median = np.median(data, axis=axis, keepdims=True)
            mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
            mad = np.where(mad == 0, 1, mad)
            normalized = (data - median) / mad

            params = {'method': 'robust', 'median': median, 'mad': mad}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    def denormalize_data(self, normalized_data: np.ndarray,
                        normalization_params: Dict[str, Any]) -> np.ndarray:
        """Denormalize data using stored parameters"""
        method = normalization_params['method']

        if method == 'standard':
            mean = normalization_params['mean']
            std = normalization_params['std']
            return normalized_data * std + mean

        elif method == 'minmax':
            min_val = normalization_params['min']
            max_val = normalization_params['max']
            return normalized_data * (max_val - min_val) + min_val

        elif method == 'robust':
            median = normalization_params['median']
            mad = normalization_params['mad']
            return normalized_data * mad + median

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def apply_transformation(self, data: Any, transformation: DataTransformation) -> Any:
        """Apply data transformation"""
        transform_type = transformation.transformation_type

        if transform_type == 'normalize':
            return self.normalize_data(data, **transformation.parameters)[0]
        elif transform_type == 'standardize':
            return self.normalize_data(data, method='standard', **transformation.parameters)[0]
        elif transform_type == 'encode_categorical':
            return self._encode_categorical(data, **transformation.parameters)
        elif transform_type == 'handle_missing':
            return self._handle_missing_values(data, **transformation.parameters)
        elif transform_type == 'feature_selection':
            return self._select_features(data, **transformation.parameters)
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")

    def _encode_categorical(self, data: pd.DataFrame, columns: List[str],
                          method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables"""
        df = data.copy()

        for col in columns:
            if method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean',
                             columns: List[str] = None) -> pd.DataFrame:
        """Handle missing values"""
        df = data.copy()
        columns = columns or df.columns

        for col in columns:
            if strategy == 'mean' and df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[col], inplace=True)

        return df

    def _select_features(self, data: np.ndarray, method: str = 'variance',
                        threshold: float = 0.01) -> np.ndarray:
        """Select important features"""
        if method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=threshold)
            return selector.fit_transform(data)
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = np.corrcoef(data.T)
            upper = np.triu(corr_matrix, k=1)
            to_drop = [i for i in range(len(upper)) if np.any(np.abs(upper[i]) > 0.95)]
            return np.delete(data, to_drop, axis=1)
        else:
            return data

    def batch_process(self, data_generator: Callable, processing_func: Callable,
                     batch_size: int = None) -> List[Any]:
        """Process data in batches for memory efficiency"""
        batch_size = batch_size or self.chunk_size
        results = []

        def process_batch(batch_data):
            return [processing_func(item) for item in batch_data]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            batch = []
            for item in data_generator():
                batch.append(item)

                if len(batch) >= batch_size:
                    future = executor.submit(process_batch, batch)
                    futures.append(future)
                    batch = []

            # Process remaining items
            if batch:
                future = executor.submit(process_batch, batch)
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())

        return results

    def create_data_pipeline(self, transformations: List[DataTransformation]) -> Callable:
        """Create a data processing pipeline"""
        def pipeline(data: Any) -> Any:
            processed_data = data
            for transform in transformations:
                processed_data = self.apply_transformation(processed_data, transform)
                self.transformations.append(transform)
            return processed_data

        return pipeline

    def optimize_memory_usage(self, data: Any) -> Any:
        """Optimize memory usage of data"""
        if isinstance(data, np.ndarray):
            # Use appropriate dtypes
            if data.dtype == np.float64 and np.all(data.astype(np.float32) == data):
                data = data.astype(np.float32)
            elif data.dtype == np.int64 and np.all(data.astype(np.int32) == data):
                data = data.astype(np.int32)

        elif isinstance(data, pd.DataFrame):
            # Downcast numeric types
            for col in data.select_dtypes(include=['float64']):
                if np.all(data[col].astype(np.float32) == data[col]):
                    data[col] = data[col].astype(np.float32)

            for col in data.select_dtypes(include=['int64']):
                if np.all(data[col].astype(np.int32) == data[col]):
                    data[col] = data[col].astype(np.int32)

        return data

    def validate_data_quality(self, data: Any) -> Dict[str, Any]:
        """Validate data quality metrics"""
        profile = self.profile_data(data)

        quality_metrics = {
            'completeness': 1.0 - (profile.null_count / np.prod(profile.shape)),
            'uniqueness': profile.unique_count / np.prod(profile.shape),
            'validity': self._check_data_validity(data),
            'consistency': self._check_data_consistency(data),
            'accuracy': self._estimate_data_accuracy(data)
        }

        return {
            'profile': profile,
            'quality_metrics': quality_metrics,
            'overall_score': np.mean(list(quality_metrics.values())),
            'recommendations': self._generate_quality_recommendations(quality_metrics)
        }

    def _check_data_validity(self, data: Any) -> float:
        """Check data validity (format, range, etc.)"""
        try:
            if isinstance(data, np.ndarray):
                # Check for infinite values
                invalid_count = np.sum(~np.isfinite(data))
                return 1.0 - (invalid_count / data.size)
            elif isinstance(data, pd.DataFrame):
                # Check for invalid values in each column
                total_cells = data.size
                invalid_cells = 0

                for col in data.columns:
                    if data[col].dtype in ['float64', 'int64']:
                        invalid_cells += data[col].isin([np.inf, -np.inf, np.nan]).sum()
                    elif data[col].dtype == 'object':
                        # Check for obviously invalid strings
                        invalid_cells += data[col].str.contains(r'^\s*$|^NULL$|^N/A$', na=False).sum()

                return 1.0 - (invalid_cells / total_cells)
            else:
                return 1.0  # Assume valid for other types
        except:
            return 0.0

    def _check_data_consistency(self, data: Any) -> float:
        """Check data consistency (relationships, constraints)"""
        try:
            if isinstance(data, pd.DataFrame):
                # Check for logical inconsistencies
                consistency_score = 1.0

                # Example: Age should be positive
                if 'age' in data.columns:
                    invalid_age = (data['age'] < 0).sum()
                    consistency_score *= 1.0 - (invalid_age / len(data))

                # Example: Salary should be positive
                if 'salary' in data.columns:
                    invalid_salary = (data['salary'] < 0).sum()
                    consistency_score *= 1.0 - (invalid_salary / len(data))

                return consistency_score
            else:
                return 1.0  # Assume consistent for other types
        except:
            return 0.0

    def _estimate_data_accuracy(self, data: Any) -> float:
        """Estimate data accuracy (ground truth comparison not available, so use heuristics)"""
        try:
            profile = self.profile_data(data)

            # Heuristic: Lower accuracy if high variance or many outliers
            if profile.statistics:
                mean = profile.statistics.get('mean', 0)
                std = profile.statistics.get('std', 1)
                cv = abs(std / mean) if mean != 0 else 1  # Coefficient of variation

                # Lower accuracy for high variability
                accuracy_penalty = min(cv / 2, 0.5)  # Max 50% penalty
                return max(0.5, 1.0 - accuracy_penalty)
            else:
                return 0.8  # Default accuracy estimate
        except:
            return 0.5

    def _generate_quality_recommendations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []

        if quality_metrics['completeness'] < 0.9:
            recommendations.append("High missing data detected. Consider imputation or data collection improvements.")

        if quality_metrics['uniqueness'] < 0.1:
            recommendations.append("Low data diversity detected. Consider collecting more varied samples.")

        if quality_metrics['validity'] < 0.95:
            recommendations.append("Data validity issues found. Check for invalid formats, ranges, or values.")

        if quality_metrics['consistency'] < 0.9:
            recommendations.append("Data consistency issues detected. Review business rules and constraints.")

        if quality_metrics['accuracy'] < 0.7:
            recommendations.append("Potential data accuracy issues. Consider validation against ground truth.")

        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring.")

        return recommendations

    def export_data_quality_report(self, validation_result: Dict[str, Any],
                                 output_path: str):
        """Export data quality report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_profile': {
                'shape': validation_result['profile'].shape,
                'data_type': validation_result['profile'].data_type.value,
                'memory_usage': validation_result['profile'].memory_usage,
                'null_count': validation_result['profile'].null_count
            },
            'quality_metrics': validation_result['quality_metrics'],
            'overall_score': validation_result['overall_score'],
            'recommendations': validation_result['recommendations']
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Data quality report exported to {output_path}")
