#!/usr/bin/env python3
"""
EEG-Based Biometric Authentication Using AMIGOS Dataset

This script presents a comprehensive study on EEG-based biometric authentication 
using the AMIGOS dataset. The research investigates the effectiveness of various 
machine learning classifiers for person identification based on EEG signal 
features extracted during emotional stimuli presentation.

Authors: [Author Name]
Institution: [Institution]
Email: [Email]

Dataset: AMIGOS - A dataset for Affect, Personality and Mood research on Individuals and Groups
Preprocessing: Bandpass filtering, artifact removal, feature extraction
Classification: Multiple ML algorithms with cross-validation
Evaluation: Accuracy, F1-score, FAR (False Acceptance Rate), FRR (False Rejection Rate)

Reproducibility Statement:
This script is designed for full reproducibility. All random seeds are set, 
dependencies are documented, and the code is structured for clarity and reusability.
"""

# =============================================================================
# 1. ENVIRONMENT SETUP AND DEPENDENCIES
# =============================================================================

import os
import pickle
import random
import time
import warnings
from itertools import islice
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import platform
import psutil

# Scientific Computing Libraries
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch, resample
from scipy.stats import skew, kurtosis, mannwhitneyu

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.base import clone

# Specialized Libraries
import antropy as entropy_analysis  # Entropy-based time series analysis
from xgboost import XGBClassifier

# Deep Learning Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Progress Tracking
from tqdm import tqdm

# =============================================================================
# 2. GLOBAL CONFIGURATION AND REPRODUCIBILITY SETUP
# =============================================================================

# Configuration Settings
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('default')
sns.set_palette("husl")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global configuration constants
EEG_SAMPLING_RATE = 128  # Hz - AMIGOS dataset sampling rate
N_JOBS = -1  # Use all available cores for parallel processing
CROSS_VALIDATION_FOLDS = 5  # Number of folds for cross-validation

def display_system_information() -> Dict[str, Any]:
    """
    Display comprehensive system information for reproducibility.
    
    Returns:
        dict: System specifications including CPU, memory, and platform details
    """
    cpu_frequency = psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    system_specs = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_frequency_mhz': cpu_frequency,
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'total_ram_gb': round(total_ram_gb, 2)
    }
    
    print("=== SYSTEM SPECIFICATIONS ===")
    for key, value in system_specs.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 30)
    
    return system_specs

# =============================================================================
# 3. DATA LOADING AND PREPROCESSING UTILITIES
# =============================================================================

def load_pickle_data(file_path: str) -> Any:
    """
    Load data from a pickle file with error handling.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        Any: Loaded data from pickle file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If there's an error loading the file
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"✓ Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading pickle file {file_path}: {str(e)}")

def save_pickle_data(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file with error handling.
    
    Args:
        data (Any): Data to be saved
        file_path (str): Path where to save the pickle file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"✓ Successfully saved data to {file_path}")
    except Exception as e:
        raise Exception(f"Error saving pickle file {file_path}: {str(e)}")

def remove_subjects_by_indices(data_dict: Dict, subject_indices_to_remove: List[int]) -> Dict:
    """
    Remove specific subjects from the dataset by their indices.
    
    Args:
        data_dict (Dict): Dictionary containing EEG data with subject keys
        subject_indices_to_remove (List[int]): List of subject indices to remove
        
    Returns:
        Dict: Updated dictionary with specified subjects removed
    """
    keys_to_remove = []
    
    for key in data_dict.keys():
        try:
            # Extract subject ID from key (assuming format like 'subject_X_...')
            subject_id = extract_subject_id_from_key(key)
            if subject_id in subject_indices_to_remove:
                keys_to_remove.append(key)
        except Exception as e:
            print(f"Warning: Could not process key {key}: {e}")
    
    # Remove identified keys
    for key in keys_to_remove:
        del data_dict[key]
        
    print(f"✓ Removed {len(keys_to_remove)} entries for subjects {subject_indices_to_remove}")
    return data_dict

def extract_subject_id_from_key(key: str) -> int:
    """
    Extract subject ID from a data key.
    
    Args:
        key (str): Data key containing subject information
        
    Returns:
        int: Subject ID extracted from the key
    """
    # Assuming key format contains subject ID - adapt based on actual format
    parts = key.split('_')
    for i, part in enumerate(parts):
        if part.lower() == 'subject' and i + 1 < len(parts):
            return int(parts[i + 1])
    
    # Alternative parsing if different format
    import re
    match = re.search(r'subject[_\s]*(\d+)', key, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Could not extract subject ID from key: {key}")

# =============================================================================
# 4. EEG SIGNAL PROCESSING AND FEATURE EXTRACTION
# =============================================================================

def calculate_bandpower_welch(eeg_signal: np.ndarray, 
                             sampling_rate: float, 
                             frequency_band: Tuple[float, float], 
                             method: str = 'welch',
                             window_duration_sec: Optional[float] = None,
                             relative_power: bool = False) -> float:
    """
    Calculate bandpower using Welch's method for EEG signal analysis.
    
    Args:
        eeg_signal (np.ndarray): EEG signal data
        sampling_rate (float): Sampling rate in Hz
        frequency_band (Tuple[float, float]): Frequency band (low, high) in Hz
        method (str): Method for power calculation ('welch')
        window_duration_sec (Optional[float]): Window duration in seconds
        relative_power (bool): Whether to return relative power
        
    Returns:
        float: Calculated bandpower
    """
    low_freq, high_freq = frequency_band
    
    # Calculate window length
    if window_duration_sec is not None:
        nperseg = int(window_duration_sec * sampling_rate)
    else:
        nperseg = min(256, len(eeg_signal))
    
    # Compute power spectral density using Welch's method
    frequencies, power_spectral_density = welch(
        eeg_signal, 
        fs=sampling_rate, 
        nperseg=nperseg
    )
    
    # Find frequency indices for the specified band
    freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    
    # Calculate bandpower
    bandpower = np.trapz(power_spectral_density[freq_mask], frequencies[freq_mask])
    
    if relative_power:
        total_power = np.trapz(power_spectral_density, frequencies)
        bandpower = bandpower / total_power
    
    return bandpower

def parse_amigos_key(data_key: str) -> Dict[str, Any]:
    """
    Parse AMIGOS dataset key to extract metadata.
    
    Args:
        data_key (str): Key from AMIGOS dataset
        
    Returns:
        Dict[str, Any]: Parsed metadata including subject, session, stimulus info
    """
    # This function should be adapted based on the actual AMIGOS key format
    # Example implementation for common formats
    parts = data_key.split('_')
    
    metadata = {
        'original_key': data_key,
        'subject_id': None,
        'session_id': None,
        'stimulus_id': None,
        'valence_level': None,
        'arousal_level': None
    }
    
    # Extract information based on key structure
    for i, part in enumerate(parts):
        if part.lower().startswith('subj'):
            metadata['subject_id'] = int(re.findall(r'\d+', part)[0])
        elif part.lower().startswith('sess'):
            metadata['session_id'] = int(re.findall(r'\d+', part)[0])
        elif part.lower().startswith('stim'):
            metadata['stimulus_id'] = int(re.findall(r'\d+', part)[0])
        elif 'valence' in part.lower():
            if 'high' in part.lower():
                metadata['valence_level'] = 'high'
            elif 'low' in part.lower():
                metadata['valence_level'] = 'low'
        elif 'arousal' in part.lower():
            if 'high' in part.lower():
                metadata['arousal_level'] = 'high'
            elif 'low' in part.lower():
                metadata['arousal_level'] = 'low'
    
    return metadata

def extract_comprehensive_eeg_features(eeg_signal: np.ndarray, 
                                     sampling_rate: float = EEG_SAMPLING_RATE) -> Dict[str, float]:
    """
    Extract comprehensive features from EEG signal for biometric authentication.
    
    Args:
        eeg_signal (np.ndarray): EEG signal data
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        Dict[str, float]: Dictionary of extracted features
    """
    features = {}
    
    # Time-domain features
    features['mean_amplitude'] = np.mean(eeg_signal)
    features['std_amplitude'] = np.std(eeg_signal)
    features['variance_amplitude'] = np.var(eeg_signal)
    features['skewness'] = skew(eeg_signal)
    features['kurtosis'] = kurtosis(eeg_signal)
    features['rms'] = np.sqrt(np.mean(eeg_signal**2))
    features['peak_to_peak'] = np.ptp(eeg_signal)
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(eeg_signal)) != 0) / len(eeg_signal)
    
    # Frequency-domain features (EEG bands)
    eeg_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    for band_name, (low_freq, high_freq) in eeg_bands.items():
        # Absolute power
        features[f'{band_name}_power'] = calculate_bandpower_welch(
            eeg_signal, sampling_rate, (low_freq, high_freq)
        )
        # Relative power
        features[f'{band_name}_relative_power'] = calculate_bandpower_welch(
            eeg_signal, sampling_rate, (low_freq, high_freq), relative_power=True
        )
    
    # Entropy-based features
    try:
        features['sample_entropy'] = entropy_analysis.sample_entropy(eeg_signal)
        features['approximate_entropy'] = entropy_analysis.app_entropy(eeg_signal)
        features['spectral_entropy'] = entropy_analysis.spectral_entropy(
            eeg_signal, sf=sampling_rate, normalize=True
        )
        features['svd_entropy'] = entropy_analysis.svd_entropy(eeg_signal, normalize=True)
    except Exception as e:
        print(f"Warning: Error calculating entropy features: {e}")
        # Set default values if entropy calculation fails
        features['sample_entropy'] = 0.0
        features['approximate_entropy'] = 0.0
        features['spectral_entropy'] = 0.0
        features['svd_entropy'] = 0.0
    
    # Additional spectral features
    frequencies, power_spectral_density = welch(eeg_signal, fs=sampling_rate)
    features['spectral_centroid'] = np.sum(frequencies * power_spectral_density) / np.sum(power_spectral_density)
    features['spectral_rolloff'] = frequencies[np.where(np.cumsum(power_spectral_density) >= 0.85 * np.sum(power_spectral_density))[0][0]]
    
    return features

def extract_features_parallel(eeg_signal: np.ndarray, 
                            sampling_rate: float = EEG_SAMPLING_RATE) -> pd.Series:
    """
    Parallel-friendly version of feature extraction that returns pandas Series.
    
    Args:
        eeg_signal (np.ndarray): EEG signal data
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        pd.Series: Series of extracted features
    """
    features_dict = extract_comprehensive_eeg_features(eeg_signal, sampling_rate)
    return pd.Series(features_dict)

# =============================================================================
# 5. DATASET PROCESSING AND FEATURE MATRIX CREATION
# =============================================================================

def process_amigos_eeg_dataset(eeg_data_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Process the AMIGOS EEG dataset and extract features for all subjects.
    
    Args:
        eeg_data_dict (Dict[str, Dict]): Raw EEG data dictionary
        
    Returns:
        Dict[str, Any]: Processed dataset with features and labels
    """
    print("Processing AMIGOS EEG dataset...")
    
    processed_data = {
        'features': {},
        'metadata': {},
        'subject_ids': set()
    }
    
    for data_key, eeg_data in tqdm(eeg_data_dict.items(), desc="Processing EEG signals"):
        try:
            # Parse key to extract metadata
            metadata = parse_amigos_key(data_key)
            subject_id = metadata['subject_id']
            
            if subject_id is None:
                print(f"Warning: Could not extract subject ID from key: {data_key}")
                continue
            
            # Extract EEG signal (assuming it's stored under 'data' key)
            if isinstance(eeg_data, dict) and 'data' in eeg_data:
                eeg_signal = eeg_data['data']
            else:
                eeg_signal = eeg_data
            
            # Ensure signal is 1D
            if eeg_signal.ndim > 1:
                # If multi-channel, take the first channel or average
                eeg_signal = np.mean(eeg_signal, axis=0)
            
            # Extract features
            features = extract_comprehensive_eeg_features(eeg_signal)
            
            # Store processed data
            processed_data['features'][data_key] = features
            processed_data['metadata'][data_key] = metadata
            processed_data['subject_ids'].add(subject_id)
            
        except Exception as e:
            print(f"Error processing {data_key}: {e}")
            continue
    
    print(f"✓ Processed {len(processed_data['features'])} EEG recordings")
    print(f"✓ Found {len(processed_data['subject_ids'])} unique subjects")
    
    return processed_data

def create_feature_dataframe(processed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a structured DataFrame from processed EEG features.
    
    Args:
        processed_data (Dict[str, Any]): Processed dataset from process_amigos_eeg_dataset
        
    Returns:
        pd.DataFrame: DataFrame with features and subject labels
    """
    feature_records = []
    
    for data_key, features in processed_data['features'].items():
        metadata = processed_data['metadata'][data_key]
        
        # Create record with features and metadata
        record = features.copy()
        record['subject_id'] = metadata['subject_id']
        record['data_key'] = data_key
        record['valence_level'] = metadata.get('valence_level', 'unknown')
        record['arousal_level'] = metadata.get('arousal_level', 'unknown')
        
        feature_records.append(record)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_records)
    
    # Set subject_id as categorical for efficient storage
    features_df['subject_id'] = features_df['subject_id'].astype('category')
    
    print(f"✓ Created feature DataFrame with shape: {features_df.shape}")
    print(f"✓ Feature columns: {len([col for col in features_df.columns if col not in ['subject_id', 'data_key', 'valence_level', 'arousal_level']])}")
    
    return features_df

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to CSV file with proper formatting.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to save
        file_path (str): Path where to save the CSV file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.to_csv(file_path, index=False, float_format='%.6f')
        print(f"✓ Successfully saved DataFrame to {file_path}")
        print(f"  Shape: {dataframe.shape}")
    except Exception as e:
        raise Exception(f"Error saving CSV file {file_path}: {str(e)}")

# =============================================================================
# 6. FEATURE SELECTION AND PREPROCESSING
# =============================================================================

def select_optimal_features(features_df: pd.DataFrame, 
                           target_column: str = 'subject_id',
                           n_features: int = 20,
                           selection_method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most informative features for biometric classification.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels
        target_column (str): Name of the target column
        n_features (int): Number of features to select
        selection_method (str): Feature selection method
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Selected features DataFrame and feature names
    """
    # Separate features from metadata
    metadata_columns = ['subject_id', 'data_key', 'valence_level', 'arousal_level']
    feature_columns = [col for col in features_df.columns if col not in metadata_columns]
    
    X = features_df[feature_columns]
    y = features_df[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Feature selection
    if selection_method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_features)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = [feature_columns[i] for i in selector.get_support(indices=True)]
    
    # Create DataFrame with selected features
    selected_features_df = pd.DataFrame(X_selected, columns=selected_feature_names)
    
    # Add metadata columns back
    for col in metadata_columns:
        if col in features_df.columns:
            selected_features_df[col] = features_df[col].values
    
    print(f"✓ Selected {len(selected_feature_names)} features using {selection_method}")
    print(f"✓ Selected features: {selected_feature_names[:5]}...")
    
    return selected_features_df, selected_feature_names

def compute_feature_statistics(features_df: pd.DataFrame, 
                              selected_features: List[str],
                              n_top_features: int = 20) -> pd.DataFrame:
    """
    Compute statistical comparison of features across subjects.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        selected_features (List[str]): List of selected feature names
        n_top_features (int): Number of top features to analyze
        
    Returns:
        pd.DataFrame: Statistical comparison results
    """
    feature_stats = []
    
    # Get unique subjects
    subjects = features_df['subject_id'].unique()
    
    for feature in selected_features[:n_top_features]:
        feature_data = features_df[feature].values
        
        # Calculate statistics across all subjects
        stats = {
            'feature_name': feature,
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'median': np.median(feature_data)
        }
        
        # Calculate between-subject variability
        subject_means = []
        for subject in subjects:
            subject_data = features_df[features_df['subject_id'] == subject][feature]
            if len(subject_data) > 0:
                subject_means.append(np.mean(subject_data))
        
        stats['between_subject_std'] = np.std(subject_means)
        stats['discriminative_power'] = stats['between_subject_std'] / (stats['std'] + 1e-8)
        
        feature_stats.append(stats)
    
    stats_df = pd.DataFrame(feature_stats)
    stats_df = stats_df.sort_values('discriminative_power', ascending=False)
    
    print(f"✓ Computed statistics for {len(feature_stats)} features")
    
    return stats_df

def preprocess_and_scale_features(features_df: pd.DataFrame, 
                                 selected_features: List[str],
                                 test_size: float = 0.2) -> Dict[str, Any]:
    """
    Preprocess features and create train/test splits with proper scaling.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features
        selected_features (List[str]): List of selected feature names
        test_size (float): Proportion of data for testing
        
    Returns:
        Dict[str, Any]: Dictionary containing preprocessed datasets
    """
    # Prepare feature matrix and labels
    X = features_df[selected_features].values
    y = features_df['subject_id'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmean(X))
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': selected_features,
        'n_classes': len(np.unique(y_encoded)),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }
    
    print(f"✓ Preprocessed data: {len(X_train)} train, {len(X_test)} test samples")
    print(f"✓ Number of classes (subjects): {preprocessed_data['n_classes']}")
    print(f"✓ Feature dimensionality: {len(selected_features)}")
    
    return preprocessed_data

# =============================================================================
# 7. BIOMETRIC EVALUATION METRICS
# =============================================================================

def calculate_false_acceptance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Acceptance Rate (FAR) for biometric systems.
    
    FAR = Number of false acceptances / Total number of impostor attempts
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: False Acceptance Rate
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate FAR: sum of false acceptances divided by total impostor attempts
    total_impostors = np.sum(cm) - np.trace(cm)  # All off-diagonal elements
    false_acceptances = total_impostors  # All misclassifications are false acceptances
    
    if total_impostors == 0:
        return 0.0
    
    far = false_acceptances / (np.sum(cm) - np.trace(cm) + np.trace(cm))
    return far

def calculate_false_rejection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Rejection Rate (FRR) for biometric systems.
    
    FRR = Number of false rejections / Total number of genuine attempts
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: False Rejection Rate
    """
    # Calculate per-class FRR and average
    unique_classes = np.unique(y_true)
    frr_per_class = []
    
    for class_label in unique_classes:
        # Genuine attempts for this class
        genuine_mask = (y_true == class_label)
        genuine_predictions = y_pred[genuine_mask]
        
        if len(genuine_predictions) == 0:
            continue
            
        # False rejections: genuine samples classified as different class
        false_rejections = np.sum(genuine_predictions != class_label)
        total_genuine = len(genuine_predictions)
        
        class_frr = false_rejections / total_genuine
        frr_per_class.append(class_frr)
    
    # Average FRR across all classes
    frr = np.mean(frr_per_class) if frr_per_class else 0.0
    return frr

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for biometric authentication.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'false_acceptance_rate': calculate_false_acceptance_rate(y_true, y_pred),
        'false_rejection_rate': calculate_false_rejection_rate(y_true, y_pred)
    }
    
    return metrics

# =============================================================================
# 8. MACHINE LEARNING CLASSIFIERS
# =============================================================================

def initialize_classifiers() -> Dict[str, Any]:
    """
    Initialize machine learning classifiers for biometric authentication.
    
    Returns:
        Dict[str, Any]: Dictionary of initialized classifiers
    """
    classifiers = {
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS
        ),
        'SupportVectorClassifier': SVC(
            kernel='rbf',
            random_state=RANDOM_SEED,
            probability=True
        ),
        'KNeighborsClassifier': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=N_JOBS
        ),
        'GradientBoostingClassifier': GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED
        ),
        'XGBoostClassifier': XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS
        )
    }
    
    print(f"✓ Initialized {len(classifiers)} classifiers")
    return classifiers

def train_and_evaluate_classifier(classifier, classifier_name: str, 
                                 X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Train and evaluate a single classifier.
    
    Args:
        classifier: Scikit-learn classifier instance
        classifier_name (str): Name of the classifier
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"Training {classifier_name}...")
    
    # Record training time
    start_time = time.time()
    
    # Train classifier
    classifier.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(
        clone(classifier), X_train, y_train, 
        cv=CROSS_VALIDATION_FOLDS, 
        scoring='accuracy',
        n_jobs=N_JOBS
    )
    
    results = {
        'classifier_name': classifier_name,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'cv_mean_accuracy': np.mean(cv_scores),
        'cv_std_accuracy': np.std(cv_scores),
        **metrics
    }
    
    print(f"✓ {classifier_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score_macro']:.4f}")
    
    return results

def train_and_evaluate_all_classifiers(preprocessed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Train and evaluate all classifiers on the preprocessed data.
    
    Args:
        preprocessed_data (Dict[str, Any]): Preprocessed dataset
        
    Returns:
        pd.DataFrame: Results for all classifiers
    """
    classifiers = initialize_classifiers()
    results = []
    
    print("\n=== TRAINING AND EVALUATION ===")
    
    for classifier_name, classifier in classifiers.items():
        try:
            result = train_and_evaluate_classifier(
                classifier, classifier_name,
                preprocessed_data['X_train'], preprocessed_data['y_train'],
                preprocessed_data['X_test'], preprocessed_data['y_test']
            )
            results.append(result)
        except Exception as e:
            print(f"✗ Error training {classifier_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print(f"\n✓ Completed evaluation of {len(results)} classifiers")
    
    return results_df

# =============================================================================
# 9. STATISTICAL ANALYSIS AND REPORTING
# =============================================================================

def perform_statistical_analysis(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical analysis on classifier results.
    
    Args:
        results_df (pd.DataFrame): Results from classifier evaluation
        
    Returns:
        Dict[str, Any]: Statistical analysis results
    """
    metrics_to_analyze = ['accuracy', 'f1_score_macro', 'false_acceptance_rate', 'false_rejection_rate']
    
    statistical_summary = {}
    
    for metric in metrics_to_analyze:
        if metric in results_df.columns:
            values = results_df[metric].values
            
            summary = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
            
            statistical_summary[metric] = summary
    
    return statistical_summary

def generate_performance_report(results_df: pd.DataFrame, 
                               statistical_summary: Dict[str, Any],
                               output_file: str = None) -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        results_df (pd.DataFrame): Results from classifier evaluation
        statistical_summary (Dict[str, Any]): Statistical analysis results
        output_file (str, optional): File path to save the report
        
    Returns:
        str: Formatted performance report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EEG BIOMETRIC AUTHENTICATION - PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Random seed: {RANDOM_SEED}")
    report_lines.append("")
    
    # Classifier rankings
    report_lines.append("CLASSIFIER PERFORMANCE RANKING (by Accuracy):")
    report_lines.append("-" * 50)
    for idx, row in results_df.iterrows():
        report_lines.append(
            f"{row['classifier_name']:30} | "
            f"Acc: {row['accuracy']:.4f} | "
            f"F1: {row['f1_score_macro']:.4f} | "
            f"FAR: {row['false_acceptance_rate']:.4f} | "
            f"FRR: {row['false_rejection_rate']:.4f}"
        )
    
    report_lines.append("")
    
    # Statistical summary
    report_lines.append("STATISTICAL SUMMARY:")
    report_lines.append("-" * 30)
    for metric, stats in statistical_summary.items():
        report_lines.append(f"\n{metric.upper()}:")
        for stat_name, value in stats.items():
            report_lines.append(f"  {stat_name:10}: {value:.6f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"✓ Performance report saved to: {output_file}")
        except Exception as e:
            print(f"✗ Error saving report: {e}")
    
    return report_text

# =============================================================================
# 10. VISUALIZATION AND PLOTTING
# =============================================================================

def create_performance_visualizations(results_df: pd.DataFrame, 
                                     output_dir: str = "./results/plots") -> None:
    """
    Create comprehensive visualizations of classifier performance.
    
    Args:
        results_df (pd.DataFrame): Results from classifier evaluation
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='accuracy', y='classifier_name', orient='h')
    plt.title('Classifier Accuracy Comparison')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1)
    
    # 2. F1-Score comparison
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x='f1_score_macro', y='classifier_name', orient='h')
    plt.title('Classifier F1-Score Comparison')
    plt.xlabel('F1-Score (Macro Average)')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_f1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Biometric metrics (FAR vs FRR)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['false_acceptance_rate'], results_df['false_rejection_rate'], 
               s=100, alpha=0.7)
    
    for idx, row in results_df.iterrows():
        plt.annotate(row['classifier_name'], 
                    (row['false_acceptance_rate'], row['false_rejection_rate']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('Biometric Performance: FAR vs FRR')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'far_frr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='training_time', y='classifier_name', orient='h')
    plt.title('Training Time Comparison')
    plt.xlabel('Training Time (seconds)')
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    # Display system information for reproducibility
    system_info = display_system_information()
    
    print(f"\n✓ EEG Biometrics Analysis Environment Ready")
    print(f"✓ Using device: {DEVICE}")
    print(f"✓ Random seed: {RANDOM_SEED}")
    print(f"✓ EEG sampling rate: {EEG_SAMPLING_RATE} Hz")
    
    # Example usage workflow (commented out - requires actual data)
    """
    # 1. Load AMIGOS dataset
    eeg_data = load_pickle_data('/path/to/amigos_eeg_data.pkl')
    
    # 2. Process dataset
    processed_data = process_amigos_eeg_dataset(eeg_data)
    
    # 3. Create feature DataFrame
    features_df = create_feature_dataframe(processed_data)
    
    # 4. Feature selection
    selected_features_df, selected_features = select_optimal_features(features_df)
    
    # 5. Preprocess and scale
    preprocessed_data = preprocess_and_scale_features(selected_features_df, selected_features)
    
    # 6. Train and evaluate classifiers
    results_df = train_and_evaluate_all_classifiers(preprocessed_data)
    
    # 7. Statistical analysis
    statistical_summary = perform_statistical_analysis(results_df)
    
    # 8. Generate report
    report = generate_performance_report(results_df, statistical_summary, './results/performance_report.txt')
    print(report)
    
    # 9. Create visualizations
    create_performance_visualizations(results_df)
    
    # 10. Save results
    results_df.to_csv('./results/classifier_results.csv', index=False)
    save_pickle_data(preprocessed_data, './results/preprocessed_data.pkl')
    """
