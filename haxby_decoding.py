import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.decomposition import PCA

def run_analysis():
    print("--- 1. Loading Haxby Dataset ---")
    haxby_dataset = datasets.fetch_haxby(subjects=[2])
    func_filename = haxby_dataset.func[0]
    mask_filename = haxby_dataset.mask_vt[0]

    labels_df = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
    conditions = labels_df["labels"]

    print("--- 2. Masking fMRI Data ---")
    masker = NiftiMasker(
        mask_img=mask_filename,
        standardize="zscore_sample",   # <- avoids deprecation
        smoothing_fwhm=4
    )
    fmri_masked = masker.fit_transform(func_filename)
    print(f"Data Shape after masking: {fmri_masked.shape} (Timepoints x Voxels)")

    print("--- 3. Filtering Conditions (Faces vs Houses) ---")
    condition_mask = conditions.isin(["face", "house"]).to_numpy()
    fmri_subset = fmri_masked[condition_mask]
    conditions_subset = conditions[condition_mask].to_numpy()

    print("--- 4. Running PCA ---")
    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(fmri_subset)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    print("--- 5. Plotting Results ---")
    plt.figure(figsize=(10, 6))
    for category in ["face", "house"]:
        m = conditions_subset == category
        plt.scatter(components[m, 0], components[m, 1], alpha=0.7, label=category)

    plt.title("PCA of Ventral Temporal Cortex Activity\nFace vs House")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_analysis()
