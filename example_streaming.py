import sys
sys.path.append(".")
import pandas as pd
import numpy as np
from EOTE.Directors import EOTEDirector
from EOTE.Builders import EoteWithMissForestInTerminalBuilder
from EOTE import StreamingEOTE


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # Step 1: Create training data (normal class only)
    # -------------------------------------------------------------------------
    print("Generating training data...")
    print("-" * 70)
    # Generate normal samples centered around specific values
    n_train = 100
    X_train = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_train),
        'feature2': np.random.normal(5, 2, n_train),
        'feature3': np.random.normal(10, 1.5, n_train),
    })
    y_train = pd.DataFrame({'class': ['normal'] * n_train})

   
    print(f"Training data shape: {X_train.shape}")
    print(f"Training data sample:\n{X_train.head()}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Build and configure EOTE
    # -------------------------------------------------------------------------
    # Use pre-built EOTE with Director
    director = EOTEDirector(EoteWithMissForestInTerminalBuilder())
    eote = director.get_eote()

    # -------------------------------------------------------------------------
    # Step 3: Create StreamingEOTE wrapper
    # -------------------------------------------------------------------------

    streaming = StreamingEOTE(
        eote=eote,
        window_size=20,                    # Process 20 samples per window
        outlier_threshold=0.0,             # Score > 0 = outlier
        drift_significance_level=0.01,     # 99% confidence for drift detection
        retraining_percentile=0.80,        # Keep bottom 80% for retraining
        min_normal_samples=5,              # Need ≥5 samples to retrain
        initial_training_required=True     # Must call train() first
    )


    # -------------------------------------------------------------------------
    # Step 4: Initial training
    # -------------------------------------------------------------------------
    streaming.train(X_train, y_train)

    # -------------------------------------------------------------------------
    # Step 5: Simulate streaming data
    # -------------------------------------------------------------------------
    print("Processing streaming data...")
    print("-" * 70)
    print()

    # Simulate 3 windows of streaming data with concept drift

    # Window 1: Normal data (similar to training)
    print("Processing Window 1 (normal data)...")
    window_1 = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 20),
        'feature2': np.random.normal(5, 2, 20),
        'feature3': np.random.normal(10, 1.5, 20),
    })

    for idx, sample in window_1.iterrows():
        result = streaming.process_sample(sample)
        if result:
            print(f"  ✓ Window {result.window_id} complete:")
            print(f"    - Total samples: {result.total_samples}")
            print(f"    - Outliers: {result.outlier_count} ({result.outlier_ratio:.1%})")
            print(f"    - Drift detected: {result.drift_detected}")
            if result.drift_detected:
                print(f"    - KS test p-value: {result.drift_p_value:.4f}")
            print(f"    - Retraining triggered: {result.retraining_triggered}")
            if result.retraining_triggered:
                print(f"    - Training samples used: {result.training_samples_used}")
    print()

    # Window 2: Concept drift (shift in feature1)
    print("Processing Window 2 (concept drift - feature1 shifted)...")
    window_2 = pd.DataFrame({
        'feature1': np.random.normal(5, 1, 20),      # Shifted from 0 to 5
        'feature2': np.random.normal(5, 2, 20),
        'feature3': np.random.normal(10, 1.5, 20),
    })

    for idx, sample in window_2.iterrows():
        result = streaming.process_sample(sample)
        if result:
            print(f"  ✓ Window {result.window_id} complete:")
            print(f"    - Total samples: {result.total_samples}")
            print(f"    - Outliers: {result.outlier_count} ({result.outlier_ratio:.1%})")
            print(f"    - Drift detected: {result.drift_detected}")
            if result.drift_detected:
                print(f"    - KS test p-value: {result.drift_p_value:.4f}")
            print(f"    - Retraining triggered: {result.retraining_triggered}")
            if result.retraining_triggered:
                print(f"    - Training samples used: {result.training_samples_used}")
                print(f"    → Model adapted to new distribution!")
    print()

    # Window 3: Continued with new distribution
    print("Processing Window 3 (continued with new distribution)...")
    window_3 = pd.DataFrame({
        'feature1': np.random.normal(5, 1, 20),      # Same as window 2
        'feature2': np.random.normal(5, 2, 20),
        'feature3': np.random.normal(10, 1.5, 20),
    })

    for idx, sample in window_3.iterrows():
        result = streaming.process_sample(sample)
        if result:
            print(f"  ✓ Window {result.window_id} complete:")
            print(f"    - Total samples: {result.total_samples}")
            print(f"    - Outliers: {result.outlier_count} ({result.outlier_ratio:.1%})")
            print(f"    - Drift detected: {result.drift_detected}")
            if result.drift_detected:
                print(f"    - KS test p-value: {result.drift_p_value:.4f}")
            print(f"    - Retraining triggered: {result.retraining_triggered}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Process batch data (alternative API)
    # -------------------------------------------------------------------------
    print("Demonstrating batch processing...")
    print("-" * 70)

    # Create batch of samples spanning multiple windows
    batch_data = pd.DataFrame({
        'feature1': np.random.normal(5, 1, 50),
        'feature2': np.random.normal(5, 2, 50),
        'feature3': np.random.normal(10, 1.5, 50),
    })

    print(f"Processing batch of {len(batch_data)} samples...")
    results = streaming.process_batch(batch_data)

    print(f"Processed {len(results)} complete windows:")
    for result in results:
        print(f"  - Window {result.window_id}: "
              f"{result.outlier_count}/{result.total_samples} outliers "
              f"({result.outlier_ratio:.1%}), "
              f"retrained: {result.retraining_triggered}")
    print()


if __name__ == '__main__':
    main()
