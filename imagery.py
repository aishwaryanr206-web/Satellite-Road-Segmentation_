from pathlib import Path

import cv2
import numpy as np
import rasterio


def stretch_to_uint8(bands):
    """Convert selected multispectral bands to displayable RGB using a robust stretch."""
    if bands.ndim != 3 or bands.shape[0] != 3:
        raise ValueError("Expected exactly three channel-first image bands.")

    rgb = np.empty((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    for index, band in enumerate(bands):
        sample = band[::max(1, band.shape[0] // 512), ::max(1, band.shape[1] // 512)]
        valid = sample[np.isfinite(sample) & (sample > 0)]
        if valid.size == 0:
            valid = sample[np.isfinite(sample)]
        if valid.size == 0:
            raise ValueError("Image band contains no finite pixel values.")
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            high = low + 1
        scaled = np.clip((band.astype(np.float32) - low) * (255 / (high - low)), 0, 255)
        rgb[:, :, index] = scaled.astype(np.uint8)
    return rgb


def read_rgb_image(path):
    """Read RGB inputs or the visible red, green, blue bands of a SpaceNet PS-MS TIFF."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            if source.count >= 5:
                bands = source.read((5, 3, 2))
                return stretch_to_uint8(bands)
            if source.count == 1:
                band = source.read(1)
                return stretch_to_uint8(np.repeat(band[None, :, :], 3, axis=0))
            if source.count >= 3:
                return stretch_to_uint8(source.read((1, 2, 3)))
            raise ValueError(f"TIFF image has unsupported band count: {source.count}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
