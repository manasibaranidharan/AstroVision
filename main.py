import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from astropy.io import fits as astropy_fits

# Extracting data from FITS File

def extract_fits(fit, patch_size=64):
    with astropy_fits.open(fit) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data
                header = hdu.header
                break
        else:
            raise ValueError("Image data not found in FITS file.")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D image, got: {data.shape}")
    
    center_y, center_x = np.array(data.shape) // 2
    half = patch_size // 2
    patch = data[center_y - half:center_y + half, center_x - half:center_x + half]

    if patch.shape != (patch_size, patch_size):
        raise ValueError("FITS image too small for requested patch size")
    
    patch = patch.astype("float32")
    patch -= np.min(patch)
    patch /= (np.max(patch) + 1e-8)
    patch = np.stack([patch] * 3, axis=-1)

    return patch, header

# Finding the distance of the object

def distance(flux, object_type):
    if flux <= 0:
        return None, None
    app_m = -2.5 * np.log10(flux) + 4.83
    M = -21.0 if object_type == 1 else 5.0
    distance_pc = 10 ** ((app_m - M + 5) / 5)
    return app_m, distance_pc

# Finding the pixel scale from CD matrix

def pixel_size(header):
    cd1_1 = header.get('CD1_1', None)
    cd1_2 = header.get('CD1_2', 0)
    cd2_1 = header.get('CD2_1', 0)
    cd2_2 = header.get('CD2_2', None)

    if cd1_1 is not None and cd2_2 is not None:
        scale_x = np.sqrt(cd1_1**2 + cd2_1**2) * 3600
        scale_y = np.sqrt(cd1_2**2 + cd2_2**2) * 3600
        return (scale_x + scale_y) / 2
    else:
        return 0.031 * 3600  

# Calculating the real size

def real_size(pixel_count, distance_pc, pixel_scale):
    angular_size_arcsec = pixel_count * pixel_scale
    real_size_pc = angular_size_arcsec * distance_pc / 206265
    return real_size_pc

# Load model

model = load_model("dataset.keras")

# Load sample FITS file

fits_path = "testdata/jw02211047001_04201_00002_nrcb2/jw02211047001_04201_00002_nrcb2_i2d.fits"

# Process FITS file

sample_patch, header = extract_fits(fits_path)
flux = np.sum(sample_patch)
pixel_scale = pixel_size(header)

# Predicting the class

pred_prob = model.predict(sample_patch[np.newaxis, ...])[0][0]
pred_class = int(pred_prob > 0.5)

# Estimating distance and size

app_m, distance_pc = distance(flux, pred_class)
object_pixel_size = 20
real_size_pc = real_size(object_pixel_size, distance_pc, pixel_scale)

# Output results

print(f"Predicted Class: {'Galaxy' if pred_class == 1 else 'Star'}")
print(f"Flux: {flux:.2f}, Apparent Mag: {app_m:.2f}, Distance: {distance_pc / 1e6:.2f} Mpc")
print(f"Size: {real_size_pc:.2f} parsecs (~{real_size_pc * 3.26:.2f} light-years)")

# Visualize

plt.imshow(sample_patch.squeeze(), cmap='gray')
plt.title(f"{'Galaxy' if pred_class else 'Star'} | Mag: {app_m:.2f} | Dist: {distance_pc / 1e6:.2f} Mpc")
plt.colorbar(label='Flux')
plt.axis('off')
plt.savefig("result.png")