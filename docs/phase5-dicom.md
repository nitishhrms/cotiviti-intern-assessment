# Phase 5 — DICOM Support + Patient Conditioning

## Goal

Accept real DICOM (.dcm) files as input — the actual format used in hospitals — and incorporate patient metadata (age, sex) as conditioning signals fed into the decoder. This demonstrates understanding of real clinical workflows and how metadata changes clinical interpretation.

## Why This Matters for Your Resume

- Every hospital system sends DICOM files, not JPEGs — supporting it signals clinical domain knowledge
- Patient conditioning (age, sex) mirrors how real radiologists adjust interpretation: a 70-year-old's "normal cardiac silhouette" differs from a 30-year-old's
- Very few academic projects touch DICOM — this is a differentiator in clinical AI interviews
- Shows understanding of HL7 FHIR / DICOM standards (critical at Epic, Philips, GE HealthCare, etc.)

---

## Prerequisites

- [ ] Phase 3 complete (FastAPI backend available)
- [ ] Install: `pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg` (for JPEG-compressed DICOMs)
- [ ] Download sample DICOM files for testing (available from TCIA: The Cancer Imaging Archive)
- [ ] Read: DICOM standard basics — windowing, photometric interpretation, pixel spacing

---

## Task 1 — DICOM File Parsing

### 1.1 — DICOM Loader
- [ ] Create `src/data/dicom_loader.py`
- [ ] Implement `load_dicom(file_path_or_bytes) -> dict`:
  - [ ] Accept both file path and raw bytes (for API upload)
  - [ ] Use `pydicom.dcmread()` to parse the file
  - [ ] Extract pixel array: `ds.pixel_array`
  - [ ] Apply rescale slope and intercept: `hu_values = pixel_array * ds.RescaleSlope + ds.RescaleIntercept`
  - [ ] Return dict with `pixel_array`, `metadata`, `patient_info`

### 1.2 — Windowing Presets
- [ ] Implement `apply_window(hu_array, window_center, window_width) -> np.ndarray`:
  - [ ] Clip HU values to `[window_center - window_width/2, window_center + window_width/2]`
  - [ ] Normalize to [0, 255] uint8
- [ ] Define standard presets:
  - [ ] **Lung window**: center=-600, width=1500 (shows lung parenchyma, pneumothorax)
  - [ ] **Mediastinum/soft tissue window**: center=40, width=400 (shows cardiac, pleural structures)
  - [ ] **Bone window**: center=400, width=1800 (shows rib fractures, vertebrae)
- [ ] Default to mediastinum window for chest X-ray report generation
- [ ] Implement `auto_window(hu_array)` that uses DICOM's own `WindowCenter` and `WindowWidth` tags if present

### 1.3 — Photometric Interpretation Handling
- [ ] Handle `PhotometricInterpretation` tag:
  - [ ] `MONOCHROME1`: high values = dark (invert the pixel values)
  - [ ] `MONOCHROME2`: high values = bright (standard)
  - [ ] `RGB` / `YBR_FULL_422`: convert to grayscale
- [ ] Handle compressed DICOMs (JPEG, JPEG 2000 transfer syntaxes) via `pylibjpeg`

### 1.4 — DICOM to Tensor Pipeline
- [ ] Implement `dicom_to_tensor(dicom_path_or_bytes, window_preset='mediastinum') -> torch.Tensor`:
  - [ ] Parse DICOM → apply windowing → normalize to 3-channel (duplicate grayscale) → resize to 224×224 → apply ImageNet normalization
  - [ ] Output: standard `(1, 3, 224, 224)` float tensor ready for the encoder
  - [ ] This is a drop-in replacement for the current PIL-based `preprocess_image()`

---

## Task 2 — Patient Metadata Extraction

### 2.1 — Metadata Parser
- [ ] Create `src/data/dicom_metadata.py`
- [ ] Implement `extract_patient_metadata(ds: pydicom.Dataset) -> dict`:
  - [ ] Extract: `PatientAge`, `PatientSex`, `StudyDate`, `Modality`, `BodyPartExamined`
  - [ ] Parse `PatientAge` (format: "045Y" → 45 or "006M" → 0.5)
  - [ ] Normalize age to float in range [0, 1]: `age / 100.0`
  - [ ] Encode sex: `{'M': 0, 'F': 1, 'O': 2, None: 2}` (0, 1, or unknown)
  - [ ] Return dict: `{'age_normalized': float, 'sex_encoded': int, 'raw': dict}`

### 2.2 — Metadata Embedding Module
- [ ] Create `src/models/metadata_encoder.py`
- [ ] Implement `MetadataEncoder(nn.Module)`:
  - [ ] Input: age (float) and sex (int)
  - [ ] Age embedding: `Linear(1, 32) → ReLU → Linear(32, 64)`
  - [ ] Sex embedding: `Embedding(3, 64)` (3 classes: M, F, Unknown)
  - [ ] Combine: concatenate → `Linear(128, 256)` to match image embedding dimension
  - [ ] Output: `(batch, 256)` metadata context vector

### 2.3 — Conditioning Integration
- [ ] Modify the image encoder output to incorporate metadata:
  - [ ] Option A (simple): Add metadata vector to image embedding (element-wise addition)
  - [ ] Option B (gated): Use a small gating network: `gate = sigmoid(Linear(512, 256))`, output = `gate * image_emb + (1-gate) * meta_emb`
  - [ ] Start with Option A; move to B if metrics improve
- [ ] Update `RadiologyReportModel.forward()` to accept optional `metadata_dict`
- [ ] Make metadata conditioning optional — if metadata is absent (JPEG upload), use zero vector

---

## Task 3 — Multi-View Support

### 3.1 — View Detection
- [ ] Implement `detect_view_position(ds) -> str`:
  - [ ] Read `ViewPosition` DICOM tag: `'PA'` (posteroanterior), `'AP'` (anteroposterior), `'LAT'` (lateral)
  - [ ] Read `SeriesDescription` if `ViewPosition` is missing and parse heuristically
  - [ ] Return normalized string: `'PA'`, `'AP'`, `'LATERAL'`, or `'UNKNOWN'`
- [ ] Add view as a metadata feature (encode as embedding: `Embedding(4, 32)`)

### 3.2 — Dual-View Fusion (Advanced)
- [ ] If both PA and LAT views are available for the same study:
  - [ ] Encode both images separately through the ViT encoder
  - [ ] Fuse embeddings: `combined = mean_pool(pa_emb, lat_emb)` (simple) or cross-attention (complex)
  - [ ] This is how real radiology AI works — PA alone misses lateral pleural pathology
- [ ] Make dual-view optional: single-view still works

---

## Task 4 — FastAPI Endpoint Update

### 4.1 — DICOM Upload Endpoint
- [ ] Add `POST /predict/dicom` endpoint to `backend/main.py`:
  - [ ] Accept `multipart/form-data` with `dicom_file: UploadFile` and optional `patient_age: float`, `patient_sex: str`
  - [ ] Parse DICOM, extract pixel array and embedded metadata
  - [ ] Override metadata with explicitly provided values if present
  - [ ] Return: `{"report": str, "pathologies": dict, "window_applied": str, "patient_info": dict, "processing_time_ms": int}`
- [ ] Update existing `POST /predict` to also accept `.dcm` file extension

### 4.2 — Preview Endpoint
- [ ] Add `POST /preview/dicom` endpoint:
  - [ ] Accept DICOM file and optional window preset name
  - [ ] Return the windowed image as JPEG bytes (for display in frontend)
  - [ ] Useful for users to verify the image was loaded correctly

---

## Task 5 — Gradio UI Update

### 5.1 — File Type Support
- [ ] Update Gradio `gr.Image` component to accept `.dcm` files
- [ ] If a DICOM is uploaded, auto-extract and display metadata in the UI

### 5.2 — Metadata Input Fields
- [ ] Add `gr.Number` input for patient age (optional, 0–120)
- [ ] Add `gr.Radio` for patient sex: "Male", "Female", "Not Specified"
- [ ] Add `gr.Dropdown` for window preset: "Auto (from DICOM)", "Lung", "Mediastinum", "Bone"

### 5.3 — Preview Panel
- [ ] Show the windowed DICOM image next to the Grad-CAM overlay
- [ ] Allow switching between window presets and re-running windowing without re-running the model

---

## Task 6 — Testing

- [ ] Create `tests/test_dicom_loader.py`
- [ ] Download 3 sample DICOM files from TCIA or RadiologyInfo.org
- [ ] Test `load_dicom()` parses correctly for each sample
- [ ] Test `apply_window()` with all three presets produces uint8 images in [0, 255]
- [ ] Test `extract_patient_metadata()` handles missing tags gracefully (returns None, not raises)
- [ ] Test `dicom_to_tensor()` output shape is `(1, 3, 224, 224)`
- [ ] Test `MetadataEncoder.forward()` output shape is `(batch, 256)`
- [ ] Test FastAPI `/predict/dicom` endpoint returns valid JSON with a DICOM test file

---

## File Structure After This Phase

```
src/
  data/
    dicom_loader.py
    dicom_metadata.py
  models/
    metadata_encoder.py
tests/
  test_dicom_loader.py
  sample_dicoms/       # 3 test DICOM files
```

---

## Definition of Done

- [ ] `POST /predict/dicom` endpoint accepts and processes real DICOM files
- [ ] Windowing correctly applied for PA chest X-rays
- [ ] Patient age and sex extracted from DICOM tags and used in model conditioning
- [ ] Gradio UI accepts DICOM uploads and shows metadata panel
- [ ] All tests pass with real sample DICOM files

---

## References

- DICOM Standard: https://www.dicomstandard.org/current
- pydicom documentation: https://pydicom.github.io
- The Cancer Imaging Archive (sample data): https://www.cancerimagingarchive.net
- Bluemke et al., "Assessing Radiology Research on Artificial Intelligence" (Radiology 2020)
