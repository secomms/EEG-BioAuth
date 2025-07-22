#!/usr/bin/env python3
"""
Example Usage Script for EEG Biometric Authentication

This script demonstrates how to use the publication-ready EEG biometrics code
with the AMIGOS dataset. It provides a complete workflow from data loading
to results generation.

Usage:
    python example_usage.py --data_path /path/to/amigos_data.pkl --output_dir ./results

Author: [Author Name]
Date: 2024
"""

import argparse
import os
import sys
from pathlib import Path

# Import our publication-ready EEG biometrics functions
from eeg_biometrics_amigos_publication_ready import *

def parse_arguments():
    """
    Parse command line arguments for the example script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='EEG Biometric Authentication Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the AMIGOS EEG dataset pickle file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results and plots'
    )
    
    parser.add_argument(
        '--n_features',
        type=int,
        default=20,
        help='Number of features to select for classification'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (0.0-1.0)'
    )
    
    parser.add_argument(
        '--remove_subjects',
        type=int,
        nargs='*',
        default=[],
        help='Subject IDs to remove from analysis (space-separated)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
        FileNotFoundError: If data file doesn't exist
    """
    # Check data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Validate test_size
    if not 0.0 < args.test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got: {args.test_size}")
    
    # Validate n_features
    if args.n_features <= 0:
        raise ValueError(f"n_features must be positive, got: {args.n_features}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print("✓ Arguments validated successfully")

def run_complete_analysis(args):
    """
    Run the complete EEG biometric analysis pipeline.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        Dict[str, Any]: Analysis results and metadata
    """
    print("=" * 80)
    print("EEG BIOMETRIC AUTHENTICATION - COMPLETE ANALYSIS")
    print("=" * 80)
    
    # Set global random seed
    global RANDOM_SEED
    RANDOM_SEED = args.random_seed
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Display system information
    system_info = display_system_information()
    
    # Step 1: Load AMIGOS dataset
    print("\n🔄 Step 1: Loading AMIGOS dataset...")
    try:
        eeg_data = load_pickle_data(args.data_path)
        print(f"✓ Loaded dataset with {len(eeg_data)} recordings")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    # Step 2: Remove specified subjects (if any)
    if args.remove_subjects:
        print(f"\n🔄 Step 2: Removing subjects {args.remove_subjects}...")
        try:
            eeg_data = remove_subjects_by_indices(eeg_data, args.remove_subjects)
            print(f"✓ Dataset now contains {len(eeg_data)} recordings")
        except Exception as e:
            print(f"✗ Error removing subjects: {e}")
            return None
    else:
        print("\n✓ Step 2: No subjects to remove, proceeding...")
    
    # Step 3: Process dataset and extract features
    print("\n🔄 Step 3: Processing EEG signals and extracting features...")
    try:
        processed_data = process_amigos_eeg_dataset(eeg_data)
        print(f"✓ Processed {len(processed_data['features'])} recordings")
        print(f"✓ Found {len(processed_data['subject_ids'])} unique subjects")
    except Exception as e:
        print(f"✗ Error processing dataset: {e}")
        return None
    
    # Step 4: Create feature DataFrame
    print("\n🔄 Step 4: Creating structured feature DataFrame...")
    try:
        features_df = create_feature_dataframe(processed_data)
        
        # Save raw features
        raw_features_path = os.path.join(args.output_dir, 'raw_features.csv')
        save_dataframe_to_csv(features_df, raw_features_path)
        
        print(f"✓ Created DataFrame with {features_df.shape[0]} samples and {features_df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error creating feature DataFrame: {e}")
        return None
    
    # Step 5: Feature selection
    print(f"\n🔄 Step 5: Selecting top {args.n_features} features...")
    try:
        selected_features_df, selected_features = select_optimal_features(
            features_df, 
            n_features=args.n_features
        )
        
        # Save selected features info
        selected_features_path = os.path.join(args.output_dir, 'selected_features.txt')
        with open(selected_features_path, 'w') as f:
            f.write("Selected Features for EEG Biometric Authentication\\n")
            f.write("=" * 50 + "\\n\\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"{i:2d}. {feature}\\n")
        
        print(f"✓ Selected {len(selected_features)} features")
    except Exception as e:
        print(f"✗ Error in feature selection: {e}")
        return None
    
    # Step 6: Compute feature statistics
    print("\n🔄 Step 6: Computing feature statistics...")
    try:
        feature_stats = compute_feature_statistics(features_df, selected_features)
        
        # Save feature statistics
        stats_path = os.path.join(args.output_dir, 'feature_statistics.csv')
        feature_stats.to_csv(stats_path, index=False, float_format='%.6f')
        
        print(f"✓ Computed statistics for {len(feature_stats)} features")
    except Exception as e:
        print(f"✗ Error computing feature statistics: {e}")
        return None
    
    # Step 7: Preprocess and scale features
    print(f"\n🔄 Step 7: Preprocessing and scaling features (test_size={args.test_size})...")
    try:
        preprocessed_data = preprocess_and_scale_features(
            selected_features_df, 
            selected_features,
            test_size=args.test_size
        )
        
        # Save preprocessing info
        preprocessing_info = {
            'n_features': len(selected_features),
            'n_classes': preprocessed_data['n_classes'],
            'n_train_samples': preprocessed_data['n_samples_train'],
            'n_test_samples': preprocessed_data['n_samples_test'],
            'test_size': args.test_size,
            'random_seed': args.random_seed
        }
        
        preprocessing_path = os.path.join(args.output_dir, 'preprocessing_info.pkl')
        save_pickle_data(preprocessing_info, preprocessing_path)
        
        print(f"✓ Preprocessed data ready for classification")
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return None
    
    # Step 8: Train and evaluate all classifiers
    print("\n🔄 Step 8: Training and evaluating classifiers...")
    try:
        results_df = train_and_evaluate_all_classifiers(preprocessed_data)
        
        # Save detailed results
        results_path = os.path.join(args.output_dir, 'classifier_results.csv')
        results_df.to_csv(results_path, index=False, float_format='%.6f')
        
        print(f"✓ Evaluated {len(results_df)} classifiers")
        
        # Display top 3 performers
        print("\\n🏆 Top 3 Performing Classifiers:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
            print(f"  {i}. {row['classifier_name']}: "
                  f"Acc={row['accuracy']:.4f}, F1={row['f1_score_macro']:.4f}")
        
    except Exception as e:
        print(f"✗ Error in classifier evaluation: {e}")
        return None
    
    # Step 9: Statistical analysis
    print("\n🔄 Step 9: Performing statistical analysis...")
    try:
        statistical_summary = perform_statistical_analysis(results_df)
        
        # Save statistical summary
        stats_summary_path = os.path.join(args.output_dir, 'statistical_summary.pkl')
        save_pickle_data(statistical_summary, stats_summary_path)
        
        print("✓ Statistical analysis completed")
    except Exception as e:
        print(f"✗ Error in statistical analysis: {e}")
        return None
    
    # Step 10: Generate comprehensive report
    print("\n🔄 Step 10: Generating performance report...")
    try:
        report_path = os.path.join(args.output_dir, 'performance_report.txt')
        report = generate_performance_report(
            results_df, 
            statistical_summary, 
            output_file=report_path
        )
        
        if args.verbose:
            print("\\n" + "=" * 50)
            print("PERFORMANCE REPORT PREVIEW:")
            print("=" * 50)
            print(report[:1000] + "..." if len(report) > 1000 else report)
        
        print("✓ Performance report generated")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return None
    
    # Step 11: Create visualizations
    print("\n🔄 Step 11: Creating performance visualizations...")
    try:
        plots_dir = os.path.join(args.output_dir, 'plots')
        create_performance_visualizations(results_df, output_dir=plots_dir)
        print("✓ Visualizations created")
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return None
    
    # Step 12: Save final processed data
    print("\n🔄 Step 12: Saving processed datasets...")
    try:
        processed_data_path = os.path.join(args.output_dir, 'preprocessed_data.pkl')
        save_pickle_data(preprocessed_data, processed_data_path)
        
        features_data_path = os.path.join(args.output_dir, 'selected_features_data.csv')
        selected_features_df.to_csv(features_data_path, index=False, float_format='%.6f')
        
        print("✓ All processed data saved")
    except Exception as e:
        print(f"✗ Error saving processed data: {e}")
        return None
    
    # Compile final results
    final_results = {
        'system_info': system_info,
        'dataset_info': {
            'original_recordings': len(eeg_data),
            'processed_recordings': len(processed_data['features']),
            'unique_subjects': len(processed_data['subject_ids']),
            'removed_subjects': args.remove_subjects
        },
        'feature_info': {
            'total_features_extracted': features_df.shape[1] - 4,  # Minus metadata columns
            'selected_features': len(selected_features),
            'feature_names': selected_features
        },
        'model_performance': {
            'best_classifier': results_df.iloc[0]['classifier_name'],
            'best_accuracy': results_df.iloc[0]['accuracy'],
            'best_f1_score': results_df.iloc[0]['f1_score_macro']
        },
        'output_files': {
            'results_csv': results_path,
            'report_txt': report_path,
            'plots_dir': plots_dir,
            'processed_data': processed_data_path
        }
    }
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Best Classifier: {final_results['model_performance']['best_classifier']}")
    print(f"🎯 Best Accuracy: {final_results['model_performance']['best_accuracy']:.4f}")
    print(f"📈 Best F1-Score: {final_results['model_performance']['best_f1_score']:.4f}")
    print(f"📁 Results saved to: {args.output_dir}")
    print("=" * 80)
    
    return final_results

def main():
    """
    Main function to run the complete EEG biometric analysis.
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_arguments(args)
        
        # Run complete analysis
        results = run_complete_analysis(args)
        
        if results is None:
            print("\\n❌ Analysis failed. Check error messages above.")
            sys.exit(1)
        
        print("\\n🎉 Analysis completed successfully!")
        print(f"📋 Check the results in: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
