<div align="center">

# 🧠 EEG-Based Biometric Authentication

### Advanced Machine Learning for Person Identification Using Brainwave Patterns

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!--
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/EEG-Biometric-Authentication/blob/main/example_usage.ipynb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yourusername/EEG-Biometric-Authentication/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/EEG-Biometric-Authentication/actions)
[![Documentation Status](https://readthedocs.org/projects/eeg-biometric-auth/badge/?version=latest)](https://eeg-biometric-auth.readthedocs.io/en/latest/?badge=latest)
-->
</div>

## 🌟 Publication-Ready Research Implementation


This repository contains an implementation of EEG-based biometric authentication. The code has been refactored for clarity, and reproducibility.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## 🔬 Overview

This research investigates the effectiveness of various machine learning classifiers for person identification based on EEG signal features extracted during emotional stimuli presentation using the AMIGOS dataset.

### Key Contributions

- **Comprehensive Feature Extraction**: Time-domain, frequency-domain, and entropy-based features
- **Multiple Classifier Evaluation**: RandomForest, LogisticRegression, SVM, KNN, GradientBoosting, XGBoost
- **Biometric-Specific Metrics**: False Acceptance Rate (FAR) and False Rejection Rate (FRR)
- **Statistical Analysis**: Comprehensive statistical evaluation with cross-validation
- **Reproducible Research**: Fixed random seeds, documented dependencies, modular code structure

## ✨ Features

### 🧠 EEG Signal Processing
- Bandpower calculation using Welch's method
- Multi-band frequency analysis (Delta, Theta, Alpha, Beta, Gamma)
- Entropy-based feature extraction (Sample, Approximate, Spectral, SVD entropy)
- Time-domain statistical features

### 🤖 Machine Learning Pipeline
- Automated feature selection using statistical methods
- Standardized preprocessing and scaling
- Cross-validation with stratified sampling
- Multiple classifier comparison
- Hyperparameter optimization ready

### 📊 Evaluation Metrics
- **Standard ML Metrics**: Accuracy, Precision, Recall, F1-Score
- **Biometric Metrics**: False Acceptance Rate (FAR), False Rejection Rate (FRR)
- **Statistical Analysis**: Mean, median, standard deviation, confidence intervals
- **Visualization**: Performance comparison plots, ROC curves, confusion matrices

### 🔄 Reproducibility Features
- Fixed random seeds across all libraries
- Comprehensive system information logging
- Version-controlled dependencies
- Modular, well-documented code structure

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for accelerated processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EEG_auth
   ```

2. **Create virtual environment**
   ```bash
   python -m venv eeg_biometrics_env
   source eeg_biometrics_env/bin/activate  # On Windows: eeg_biometrics_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python eeg_biometrics_amigos.py
   ```

## 📖 Usage

### Basic Usage

```python
from eeg_biometrics_amigos_publication_ready import *

# 1. Load your AMIGOS dataset
eeg_data = load_pickle_data('/path/to/amigos_eeg_data.pkl')

# 2. Process the dataset
processed_data = process_amigos_eeg_dataset(eeg_data)

# 3. Create feature DataFrame
features_df = create_feature_dataframe(processed_data)

# 4. Feature selection
selected_features_df, selected_features = select_optimal_features(features_df)

# 5. Preprocess and scale features
preprocessed_data = preprocess_and_scale_features(selected_features_df, selected_features)

# 6. Train and evaluate all classifiers
results_df = train_and_evaluate_all_classifiers(preprocessed_data)

# 7. Generate comprehensive report
statistical_summary = perform_statistical_analysis(results_df)
report = generate_performance_report(results_df, statistical_summary)

# 8. Create visualizations
create_performance_visualizations(results_df)
```

### Advanced Configuration

```python
# Custom feature selection
selected_features_df, selected_features = select_optimal_features(
    features_df, 
    n_features=50,  # Select top 50 features
    selection_method='f_classif'
)

# Custom preprocessing
preprocessed_data = preprocess_and_scale_features(
    features_df, 
    selected_features,
    test_size=0.3  # 30% for testing
)

# Individual classifier training
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
results = train_and_evaluate_classifier(
    classifier, 'CustomRandomForest',
    preprocessed_data['X_train'], preprocessed_data['y_train'],
    preprocessed_data['X_test'], preprocessed_data['y_test']
)
```

## 📊 Dataset

### AMIGOS Dataset
The AMIGOS dataset contains EEG recordings from participants watching emotional video stimuli. 

**Dataset Structure Expected:**
```
amigos_data.pkl
├── subject_1_session_1_stimulus_1: {data: np.array, metadata: dict}
├── subject_1_session_1_stimulus_2: {data: np.array, metadata: dict}
└── ...
```

**Key Information:**
- **Participants**: Multiple subjects
- **Sampling Rate**: 128 Hz
- **Channels**: EEG channels (adaptable to multi-channel)
- **Stimuli**: Emotional video clips with valence/arousal labels

### Data Preprocessing
1. **Signal Cleaning**: Artifact removal and filtering
2. **Feature Extraction**: 30+ features per signal segment
3. **Normalization**: Z-score standardization
4. **Label Encoding**: Subject ID encoding for classification

## 🔬 Methodology

### Feature Extraction Pipeline

1. **Time-Domain Features**
   - Mean, Standard Deviation, Variance
   - Skewness, Kurtosis
   - RMS, Peak-to-Peak amplitude
   - Zero-crossing rate

2. **Frequency-Domain Features**
   - Bandpower for EEG frequency bands (δ, θ, α, β, γ)
   - Relative power ratios
   - Spectral centroid and rolloff

3. **Entropy-Based Features**
   - Sample Entropy
   - Approximate Entropy
   - Spectral Entropy
   - Singular Value Decomposition Entropy

### Classification Pipeline

1. **Feature Selection**: Statistical significance testing (F-test)
2. **Preprocessing**: Standardization and scaling
3. **Model Training**: 6 different ML algorithms
4. **Evaluation**: 5-fold cross-validation
5. **Metrics**: Comprehensive biometric evaluation

### Classifiers Evaluated

| Classifier | Key Parameters | Use Case |
|------------|----------------|----------|
| Random Forest | n_estimators=100, max_depth=10 | Ensemble learning |
| Logistic Regression | max_iter=1000, L2 regularization | Linear baseline |
| Support Vector Machine | RBF kernel, probability=True | Non-linear separation |
| K-Nearest Neighbors | k=5, distance-weighted | Instance-based learning |
| Gradient Boosting | n_estimators=100, learning_rate=0.1 | Boosting ensemble |
| XGBoost | n_estimators=100, optimized | Advanced boosting |

## 📈 Results

### Expected Performance Metrics

The system evaluates classifiers using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **FAR**: False Acceptance Rate (security metric)
- **FRR**: False Rejection Rate (usability metric)

### Output Files Generated

```
results/
├── classifier_results.csv          # Detailed results table
├── performance_report.txt          # Comprehensive text report
├── preprocessed_data.pkl          # Processed dataset
└── plots/
    ├── accuracy_f1_comparison.png  # Performance comparison
    ├── far_frr_comparison.png      # Biometric metrics plot
    └── training_time_comparison.png # Efficiency analysis
```

## 🔄 Reproducibility

### Reproducibility Features

1. **Fixed Random Seeds**: All random operations use seed=42
2. **Version Control**: Exact package versions specified
3. **System Logging**: Hardware and software specifications recorded
4. **Modular Design**: Clear separation of concerns
5. **Comprehensive Documentation**: Every function documented

### Environment Information

The system automatically logs:
- Operating system and Python version
- CPU specifications and memory
- GPU availability and specifications
- Package versions and dependencies
- Execution timestamp

### Verification

To verify reproducibility:

```bash
# Run the same experiment multiple times
python eeg_biometrics_amigos_publication_ready.py
# Results should be identical across runs
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{eeg_biometrics_amigos_2024,
  title={EEG-Based Biometric Authentication Using AMIGOS Dataset: A Comprehensive Machine Learning Approach},
  author={[Author Name]},
  journal={[Journal Name]},
  year={2024},
  publisher={[Publisher]},
  doi={[DOI]},
  url={[URL]}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Ensure reproducibility is maintained

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AMIGOS Dataset**: Thanks to the creators of the AMIGOS dataset
- **Scientific Community**: Built on open-source scientific Python ecosystem
- **Contributors**: All contributors to this research

## 📞 Contact

For questions, issues, or collaborations:

- **Email**: [your.email@institution.edu]
- **GitHub Issues**: [Repository Issues Page]
- **Research Gate**: [Your Research Gate Profile]

---

## 🔧 Technical Details

### System Requirements

**Minimum Requirements:**
- RAM: 8GB
- CPU: Multi-core processor
- Storage: 2GB free space
- Python: 3.8+

**Recommended Requirements:**
- RAM: 16GB+
- CPU: 8+ cores
- GPU: CUDA-compatible (optional)
- Storage: 5GB+ free space
- Python: 3.9+

### Performance Optimization

The code includes several optimization features:

1. **Parallel Processing**: Multi-core utilization where possible
2. **Memory Efficiency**: Optimized data structures
3. **GPU Support**: CUDA acceleration for compatible operations
4. **Caching**: Intermediate results caching for repeated runs

### Troubleshooting

**Common Issues:**

1. **Memory Errors**: Reduce batch size or feature count
2. **CUDA Errors**: Ensure proper GPU drivers and PyTorch installation
3. **Import Errors**: Verify all dependencies are installed
4. **Data Format Errors**: Check AMIGOS dataset format compatibility

**Getting Help:**

1. Check the [Issues](issues) page for known problems
2. Review the documentation thoroughly
3. Contact the maintainers for research-specific questions

---

*This README was generated as part of a publication-ready research package. Last updated: 2024*
