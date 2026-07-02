"""Extract and normalize patient metadata from a pydicom Dataset."""

import re


def _parse_age(age_str) -> float:
    """
    Convert DICOM PatientAge string to a float in years.

    DICOM age format examples: '045Y', '006M', '002W', '010D'
    Returns 0.0 if unparseable.
    """
    if age_str is None:
        return None
    age_str = str(age_str).strip().upper()
    match = re.match(r"(\d+)([YMWD]?)", age_str)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit in ("", "Y"):
        return float(value)
    elif unit == "M":
        return value / 12.0
    elif unit == "W":
        return value / 52.0
    elif unit == "D":
        return value / 365.0
    return float(value)


_SEX_MAP = {"M": 0, "F": 1, "O": 2, None: 2, "": 2}


def extract_patient_metadata(ds) -> dict:
    """
    Extract and normalize patient metadata from a pydicom Dataset.

    Returns:
        dict with keys:
            age_normalized (float [0,1] or None),
            sex_encoded    (int: 0=M, 1=F, 2=Unknown),
            raw            (dict of raw tag values)
    """
    raw_age = getattr(ds, "PatientAge", None)
    raw_sex = str(getattr(ds, "PatientSex", "") or "").upper().strip()

    age_years = _parse_age(raw_age)
    age_normalized = min(age_years / 100.0, 1.0) if age_years is not None else None

    sex_encoded = _SEX_MAP.get(raw_sex, 2)

    return {
        "age_normalized": age_normalized,
        "sex_encoded": sex_encoded,
        "raw": {
            "PatientAge": str(raw_age) if raw_age else None,
            "PatientSex": raw_sex or None,
            "StudyDate": str(getattr(ds, "StudyDate", None) or ""),
            "Modality": str(getattr(ds, "Modality", None) or ""),
            "BodyPartExamined": str(getattr(ds, "BodyPartExamined", None) or ""),
        },
    }
