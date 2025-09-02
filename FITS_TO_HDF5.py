import numpy as np
import h5py
from astropy.io import fits
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval
import os

def fits_to_hdf5(fits_path, patch_size=64, step=64, resize_to=None, output_dir=".", test_ratio=0.1, val_ratio=0.1, use_rgb=False):
    os.makedirs(output_dir, exist_ok=True)

    with fits.open(fits_path) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        data = np.nan_to_num(data)

    # Normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    patches = []
    labels = []
    h, w = data.shape

    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patch = data[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size):
                if resize_to:
                    patch = resize(patch, resize_to, preserve_range=True, anti_aliasing=True)
                patch = patch.astype(np.float32)

                if use_rgb:
                    # convert grayscale to fake RGB
                    patch = np.stack([patch]*3, axis=-1)  
                else:
                    # grayscale with 1 channel
                    patch = patch[..., np.newaxis]  

                patches.append(patch)
                labels.append(1 if np.mean(patch) > 0.2 else 0)

    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels, dtype=np.uint8)

    print(f"Total patches generated: {len(patches)}")

    if len(patches) < 5:
        print("Not enough samples to perform train/val/test split.")
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = patches, labels
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            patches, labels, test_size=test_ratio + val_ratio, random_state=42
        )
        val_portion = val_ratio / (test_ratio + val_ratio)

        if len(X_temp) < 2:
            print("Not enough temp samples to split into val/test. All go to test.")
            X_val, y_val = [], []
            X_test, y_test = X_temp, y_temp
        else:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1 - val_portion, random_state=42
            )

    def save_to_hdf5(X, y, name):
        if len(X) == 0: return
        h5_path = os.path.join(output_dir, f"{name}.h5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, "a") as f:
                original_len = f["X"].shape[0]
                new_len = original_len + X.shape[0]

                f["X"].resize((new_len,) + f["X"].shape[1:])
                f["y"].resize((new_len,))

                f["X"][original_len:] = X
                f["y"][original_len:] = y
            print(f"Appended to {h5_path} (now X: {new_len}, added: {X.shape[0]})")
        else:
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("X", data=X, maxshape=(None,) + X.shape[1:], compression="gzip")
                f.create_dataset("y", data=y, maxshape=(None,), compression="gzip")
            print(f"Created {h5_path} (X: {X.shape}, y: {y.shape})")

    save_to_hdf5(X_train, y_train, "train")
    save_to_hdf5(X_val, y_val, "val")
    save_to_hdf5(X_test, y_test, "test")

    print("Conversion complete:")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

directory_path = './raw_data/M31/'
files_and_dirs = os.listdir(directory_path)

for item in files_and_dirs:
    full_path = os.path.join(directory_path, item)
    if os.path.isfile(full_path):  
        fits_to_hdf5(
        full_path,
        patch_size=64,
        resize_to=(64, 64),
        step=64,
        use_rgb=True,
        output_dir="Train/dataset/."
)
