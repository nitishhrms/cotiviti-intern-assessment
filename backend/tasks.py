from backend.celery_app import celery_app


@celery_app.task(bind=True, name="backend.tasks.run_inference")
def run_inference(self, image_bytes_hex: str) -> dict:
    from backend.inference import engine

    image_bytes = bytes.fromhex(image_bytes_hex)
    try:
        result = engine.predict(image_bytes)
        return result
    except Exception as exc:
        self.retry(exc=exc, countdown=2, max_retries=2)


@celery_app.task(bind=True, name="backend.tasks.run_dicom_inference")
def run_dicom_inference(self, dicom_bytes_hex: str, window_preset: str = "mediastinum") -> dict:
    from backend.inference import engine
    from src.data.dicom_loader import dicom_to_tensor

    dicom_bytes = bytes.fromhex(dicom_bytes_hex)
    try:
        tensor, patient_info, preset_used = dicom_to_tensor(dicom_bytes, window_preset)
        result = engine.predict_from_tensor(tensor)
        result["window_applied"] = preset_used
        result["patient_info"] = patient_info
        return result
    except Exception as exc:
        self.retry(exc=exc, countdown=2, max_retries=2)
