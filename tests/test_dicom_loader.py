"""
Unit tests for DICOM loader and metadata extractor.

Tests that do not require real DICOM files use synthetic data.
Tests marked with @pytest.mark.skipif skip gracefully when pydicom is absent.
"""

import io
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_dicom_dataset(age="045Y", sex="M"):
    """Return a minimal mock pydicom Dataset."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
    except ImportError:
        pytest.skip("pydicom not installed")

    ds = Dataset()
    ds.PatientAge = age
    ds.PatientSex = sex
    ds.Modality = "CR"
    ds.BodyPartExamined = "CHEST"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.Rows = 64
    ds.Columns = 64
    arr = np.random.randint(0, 2048, (64, 64), dtype=np.uint16)
    ds.PixelData = arr.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    return ds, arr


# ---------------------------------------------------------------------------
# dicom_metadata tests
# ---------------------------------------------------------------------------

def test_extract_patient_metadata_normal():
    try:
        import pydicom
    except ImportError:
        pytest.skip("pydicom not installed")

    from src.data.dicom_metadata import extract_patient_metadata
    ds, _ = _make_fake_dicom_dataset(age="050Y", sex="F")
    meta = extract_patient_metadata(ds)

    assert meta["age_normalized"] == pytest.approx(0.5, abs=1e-4)
    assert meta["sex_encoded"] == 1  # F


def test_extract_patient_metadata_missing_fields():
    try:
        import pydicom
        from pydicom.dataset import Dataset
    except ImportError:
        pytest.skip("pydicom not installed")

    from src.data.dicom_metadata import extract_patient_metadata
    ds = pydicom.Dataset()
    meta = extract_patient_metadata(ds)
    assert meta["age_normalized"] is None
    assert meta["sex_encoded"] == 2  # unknown


def test_extract_patient_metadata_month_age():
    try:
        import pydicom
    except ImportError:
        pytest.skip("pydicom not installed")

    from src.data.dicom_metadata import extract_patient_metadata
    ds, _ = _make_fake_dicom_dataset(age="006M", sex="M")
    meta = extract_patient_metadata(ds)
    assert meta["age_normalized"] == pytest.approx(0.5 / 100, abs=0.01)


# ---------------------------------------------------------------------------
# dicom_loader tests
# ---------------------------------------------------------------------------

def test_apply_window_output_range():
    from src.data.dicom_loader import apply_window
    arr = np.linspace(-2000, 2000, 1000).astype(np.float32)
    result = apply_window(arr, window_center=40, window_width=400)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_apply_window_all_presets():
    from src.data.dicom_loader import apply_window, WINDOW_PRESETS
    arr = np.random.uniform(-1000, 3000, (64, 64)).astype(np.float32)
    for name, params in WINDOW_PRESETS.items():
        result = apply_window(arr, params["center"], params["width"])
        assert result.shape == (64, 64), f"Preset {name} produced wrong shape"
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# MetadataEncoder tests
# ---------------------------------------------------------------------------

def test_metadata_encoder_output_shape():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from src.models.metadata_encoder import MetadataEncoder
    encoder = MetadataEncoder(embed_dim=256)
    encoder.eval()

    import torch
    age = torch.tensor([0.45, 0.70], dtype=torch.float32)
    sex = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        out = encoder(age, sex)

    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"
