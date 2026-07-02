import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


WINDOW_PRESETS = {
    "lung":        {"center": -600, "width": 1500},
    "mediastinum": {"center":   40, "width":  400},
    "bone":        {"center":  400, "width": 1800},
}

_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def apply_window(hu_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    clipped = np.clip(hu_array.astype(np.float32), lo, hi)
    normalized = (clipped - lo) / (hi - lo) * 255.0
    return normalized.astype(np.uint8)


def auto_window(ds) -> dict:
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return WINDOW_PRESETS["mediastinum"]
    if hasattr(wc, "__iter__") and not isinstance(wc, (int, float)):
        wc = float(wc[0])
    if hasattr(ww, "__iter__") and not isinstance(ww, (int, float)):
        ww = float(ww[0])
    return {"center": float(wc), "width": float(ww)}


def load_dicom(file_path_or_bytes) -> dict:
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")

    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        ds = pydicom.dcmread(io.BytesIO(file_path_or_bytes))
    else:
        ds = pydicom.dcmread(file_path_or_bytes)

    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    from src.data.dicom_metadata import extract_patient_metadata
    patient_info = extract_patient_metadata(ds)

    metadata = {
        "StudyDate": str(getattr(ds, "StudyDate", "")),
        "Modality": str(getattr(ds, "Modality", "")),
        "BodyPartExamined": str(getattr(ds, "BodyPartExamined", "")),
        "PhotometricInterpretation": photometric,
        "Rows": int(getattr(ds, "Rows", 0)),
        "Columns": int(getattr(ds, "Columns", 0)),
        "ViewPosition": str(getattr(ds, "ViewPosition", "UNKNOWN")),
    }

    return {
        "pixel_array": pixel_array,
        "metadata": metadata,
        "patient_info": patient_info,
        "ds": ds,
    }


def dicom_to_tensor(
    dicom_path_or_bytes,
    window_preset: str = "mediastinum",
) -> tuple:
    result = load_dicom(dicom_path_or_bytes)
    pixel_array = result["pixel_array"]
    ds = result["ds"]

    if window_preset == "auto":
        window_params = auto_window(ds)
        preset_name = "auto"
    else:
        window_params = WINDOW_PRESETS.get(window_preset, WINDOW_PRESETS["mediastinum"])
        preset_name = window_preset

    windowed = apply_window(pixel_array, window_params["center"], window_params["width"])

    if windowed.ndim == 3:
        windowed = windowed[0]

    pil_image = Image.fromarray(windowed, mode="L").convert("RGB")

    tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _IMAGENET_NORMALIZE,
    ])(pil_image).unsqueeze(0)

    return tensor, result["patient_info"], preset_name


def detect_view_position(ds) -> str:
    vp = str(getattr(ds, "ViewPosition", "")).upper().strip()
    if vp in ("PA", "AP"):
        return vp
    if "LAT" in vp:
        return "LATERAL"
    sd = str(getattr(ds, "SeriesDescription", "")).upper()
    if "PA" in sd:
        return "PA"
    if "AP" in sd:
        return "AP"
    if "LAT" in sd:
        return "LATERAL"
    return "UNKNOWN"
