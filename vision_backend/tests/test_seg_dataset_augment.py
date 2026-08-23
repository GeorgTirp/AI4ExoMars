"""Tests for seg_dataset.py's new augmentation (spatial/photometric jitter) and
compute_class_pixel_counts (class-imbalance mitigation support)."""

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

from vision_backend.seg_dataset import (
    SegCropRecord,
    SegmentationCropDataset,
    compute_class_pixel_counts,
)

IGNORE_INDEX = 255


@pytest.fixture
def paired_rasters(tmp_path):
    """A 256x256 imagery+label pair: left half class 1, right half class 2,
    a small nodata stripe down the middle, matching col/row/transform exactly
    (so col,row addressing behaves as in the real aligned pipeline)."""
    size = 256
    imagery = np.full((size, size), 150, dtype=np.uint8)
    imagery[:, 120:136] = 0  # nodata stripe

    labels = np.zeros((size, size), dtype=np.uint8)
    labels[:, :128] = 1  # DC id 1 -> class index 0
    labels[:, 128:] = 2  # DC id 2 -> class index 1

    transform = Affine.identity()
    imagery_path = tmp_path / "imagery.tif"
    label_path = tmp_path / "labels.tif"
    for path, data in ((imagery_path, imagery), (label_path, labels)):
        with rasterio.open(
            str(path), "w", driver="GTiff", height=size, width=size, count=1,
            dtype=np.uint8, crs="EPSG:4326", transform=transform, nodata=0,
        ) as dst:
            dst.write(data, 1)

    return imagery_path, label_path


def test_no_augment_is_deterministic(paired_rasters):
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=140, row=32, size=64, split="train")]
    ds = SegmentationCropDataset(records, imagery_path=imagery_path, label_path=label_path, augment=False)

    first = ds[0]
    second = ds[0]
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["label"], second["label"])


def test_crop_without_augment_reads_expected_class(paired_rasters):
    imagery_path, label_path = paired_rasters
    # Entirely in the right half (class index 1 = DC id 2), no nodata stripe.
    records = [SegCropRecord(col=160, row=32, size=64, split="train")]
    ds = SegmentationCropDataset(records, imagery_path=imagery_path, label_path=label_path, augment=False)

    sample = ds[0]
    assert torch.all(sample["label"] == 1)


def test_spatial_jitter_varies_crop_position(paired_rasters):
    imagery_path, label_path = paired_rasters
    # Straddling the class boundary (col=128) so a jittered crop's label
    # composition visibly shifts depending on the random offset drawn.
    records = [SegCropRecord(col=96, row=32, size=64, split="train")]
    ds = SegmentationCropDataset(
        records, imagery_path=imagery_path, label_path=label_path,
        augment=True, spatial_jitter_px=20, brightness_jitter=0, contrast_jitter=0,
    )

    torch.manual_seed(0)
    class0_fractions = {float((ds[0]["label"] == 0).float().mean()) for _ in range(20)}
    # If jitter did nothing, every draw would land on exactly the same fraction.
    assert len(class0_fractions) > 1


def test_zero_jitter_matches_unaugmented_position(paired_rasters):
    """spatial_jitter_px=0 with augment=True still flips, but crop position
    should behave the same as augment=False (no positional drift)."""
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=160, row=32, size=64, split="train")]
    ds = SegmentationCropDataset(
        records, imagery_path=imagery_path, label_path=label_path,
        augment=True, spatial_jitter_px=0, brightness_jitter=0, contrast_jitter=0,
    )
    for _ in range(10):
        sample = ds[0]
        assert torch.all(sample["label"] == 1)  # always fully inside class-index-1 region


def test_photometric_jitter_changes_valid_pixel_values_but_not_invalid(paired_rasters):
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=0, row=0, size=256, split="train")]  # spans the nodata stripe
    ds = SegmentationCropDataset(
        records, imagery_path=imagery_path, label_path=label_path,
        augment=True, spatial_jitter_px=0, brightness_jitter=0.5, contrast_jitter=0.5,
    )

    torch.manual_seed(1)
    seen_values = set()
    for _ in range(20):
        img = ds[0]["image"]
        invalid_mask = img[0] == -1.0
        # The nodata stripe columns must stay exactly -1.0 regardless of jitter.
        assert torch.all(img[0, :, 120:136] == -1.0)
        seen_values.add(round(float(img[0, 10, 10]), 4))
    # Valid-pixel value should have varied across draws (brightness/contrast jitter active).
    assert len(seen_values) > 1


def test_photometric_jitter_disabled_when_zero(paired_rasters):
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=160, row=32, size=64, split="train")]
    ds = SegmentationCropDataset(
        records, imagery_path=imagery_path, label_path=label_path,
        augment=True, spatial_jitter_px=0, brightness_jitter=0.0, contrast_jitter=0.0,
    )
    baseline = ds[0]["image"]
    for _ in range(5):
        assert torch.equal(ds[0]["image"], baseline)


def test_val_style_augment_false_ignores_jitter_params(paired_rasters):
    """augment=False must fully disable jitter/photometric even if configured
    with nonzero magnitudes -- this is how val datasets stay unaugmented."""
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=160, row=32, size=64, split="val")]
    ds = SegmentationCropDataset(
        records, imagery_path=imagery_path, label_path=label_path,
        augment=False, spatial_jitter_px=50, brightness_jitter=0.9, contrast_jitter=0.9,
    )
    baseline = ds[0]["image"]
    for _ in range(10):
        assert torch.equal(ds[0]["image"], baseline)


def test_compute_class_pixel_counts_matches_known_layout(paired_rasters):
    imagery_path, label_path = paired_rasters
    # Two disjoint crops, one per class half, no nodata stripe inside either.
    records = [
        SegCropRecord(col=0, row=0, size=64, split="train"),      # class index 0
        SegCropRecord(col=160, row=0, size=64, split="train"),    # class index 1
    ]
    counts = compute_class_pixel_counts(
        records, imagery_path=imagery_path, label_path=label_path,
        num_classes=2, ignore_index=IGNORE_INDEX,
    )
    assert counts == {0: 64 * 64, 1: 64 * 64}


def test_flip_augment_joint_consistency_across_many_seeds(paired_rasters):
    """F6 verify: a flip (h and/or v) must be applied identically to image and
    label -- checked here via a fixed per-pixel img<->label correspondence
    that only survives if the same spatial transform hit both arrays."""
    imagery_path, label_path = paired_rasters
    records = [SegCropRecord(col=0, row=0, size=64, split="train")]
    ds = SegmentationCropDataset(records, imagery_path=imagery_path, label_path=label_path, augment=False)

    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    lab_offset = 1000
    lab = (img + lab_offset).astype(np.int64)

    seen_transforms = set()
    for seed in range(30):
        torch.manual_seed(seed)
        flipped_img, flipped_lab = ds._flip_augment(img.copy(), lab.copy())
        assert np.array_equal(flipped_lab, flipped_img.astype(np.int64) + lab_offset)
        seen_transforms.add(flipped_img.tobytes())
    assert len(seen_transforms) > 1  # confirms flips vary across draws, not a no-op


def test_no_rotation_in_augmentation_pipeline():
    """F6 verify: rotation is forbidden (HiRISE shading direction is fixed by
    sun azimuth -- see seg_dataset.py's module docstring); only flips/jitter
    are permitted spatial augmentation."""
    import inspect

    from vision_backend import seg_dataset

    source = inspect.getsource(seg_dataset.SegmentationCropDataset)
    assert "rot90" not in source
    assert ".rotate(" not in source
    assert "np.rot90" not in source


def test_compute_class_pixel_counts_excludes_imagery_nodata(paired_rasters):
    imagery_path, label_path = paired_rasters
    # Crop spans columns [64, 128); the nodata stripe [120,136) only overlaps
    # it in [120, 128) -- 8 columns, not the full 16-wide stripe.
    records = [SegCropRecord(col=64, row=0, size=64, split="train")]
    counts = compute_class_pixel_counts(
        records, imagery_path=imagery_path, label_path=label_path,
        num_classes=2, ignore_index=IGNORE_INDEX,
    )
    nodata_cols_in_crop = 8  # [120, 128)
    expected_class0 = 64 * 64 - 64 * nodata_cols_in_crop
    assert counts[0] == expected_class0
    assert counts[1] == 0
