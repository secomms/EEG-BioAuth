# Standard Library Imports
import os
import pickle
import random
import time
import warnings
from itertools import islice
from typing import Dict, List, Tuple, Any, Optional

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

# Configuration Settings
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('seaborn-v0_8')
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
print(f"Using device: {DEVICE}")

# Global configuration constants
EEG_SAMPLING_RATE = 128  # Hz - AMIGOS dataset sampling rate
N_JOBS = -1  # Use all available cores for parallel processing

print("✓ All libraries imported successfully")
print(f"✓ Random seed set to: {RANDOM_SEED}")
print(f"✓ EEG sampling rate: {EEG_SAMPLING_RATE} Hz")
