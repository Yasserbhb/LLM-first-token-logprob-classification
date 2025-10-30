"""
Quick script to check what labels are actually in the extracted data
"""

import numpy as np
from collections import Counter

# Try to load the labels from the results if saved
try:
    import pickle
    import joblib

    # Check if there are any saved results
    import os
    files = [f for f in os.listdir('.') if f.startswith('model_metadata_') and f.endswith('_8bit.pkl')]

    if files:
        print("="*80)
        print("CHECKING SAVED MODEL LABELS")
        print("="*80)

        for file in files:
            print(f"\nFile: {file}")
            with open(file, 'rb') as f:
                metadata = pickle.load(f)
                print(f"  Model: {metadata.get('model_name', 'Unknown')}")

        # Try to load a result object if it exists
        result_files = [f for f in os.listdir('.') if f.startswith('boilerplate_detector_') and f.endswith('_8bit.pkl')]
        if result_files:
            # Just check the first one
            print(f"\n\nChecking actual k-NN training data from: {result_files[0]}")
            knn = joblib.load(result_files[0])

            if hasattr(knn, 'y_train') and knn.y_train is not None:
                labels = knn.y_train
                print(f"\nUnique labels found: {sorted(np.unique(labels))}")
                print(f"Total unique labels: {len(np.unique(labels))}")

                print(f"\nLabel distribution:")
                for label, count in Counter(labels).most_common():
                    print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
            else:
                print("  No training labels found in k-NN model")

    else:
        print("No saved model files found. Please run the experiment first.")

except Exception as e:
    print(f"Error loading saved data: {e}")
    print("\nPlease run the notebook first to generate the data.")

# Also check the dataset directly
print("\n" + "="*80)
print("CHECKING ORIGINAL DATASET")
print("="*80)

try:
    from datasets import load_dataset

    print("\nLoading dataset...")
    dataset = load_dataset("jfrog/boilerplate-detection")

    labels = [ex['type'] for ex in dataset['train']]
    unique_labels = sorted(set(labels))

    print(f"\nUnique labels in dataset: {unique_labels}")
    print(f"Total unique labels: {len(unique_labels)}")

    print(f"\nLabel distribution:")
    for label, count in Counter(labels).most_common():
        print(f"  '{label}': {count} ({count/len(labels)*100:.1f}%)")
        # Check for any weird characters
        print(f"    -> Length: {len(label)}, Repr: {repr(label)}")

except Exception as e:
    print(f"Error loading dataset: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
