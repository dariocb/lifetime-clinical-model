"""Optional pretrained embeddings of partial baseline EHR notes."""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def embed_ehr(dataset, model_name=DEFAULT_MODEL, batch_size=32, device="cpu", local_files_only=False):
    """Mutate dataset with patient-aligned normalized vectors and provenance.

    First call downloads model weights unless cached/local_files_only. Missing
    notes get zero vectors and retain ehr_missing=True, never invented text.
    No trainable clinical component is implemented here.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError('Install embeddings with: pip install -e ".[embeddings]"') from exc
    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    notes = dataset.patients.ehr_history
    valid = notes.notna() & notes.fillna("").str.strip().ne("")
    vectors = np.zeros((len(notes),model.get_sentence_embedding_dimension()),dtype=np.float32)
    if valid.any():
        vectors[valid.to_numpy()] = model.encode(notes[valid].tolist(),batch_size=batch_size,
                   normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    dataset.ehr_embeddings = vectors
    dataset.embedding_metadata = {"model":model_name,"dimension":vectors.shape[1],
            "normalized":True,"missing_policy":"zero vector plus ehr_missing mask",
            "row_order":"patients.patient_id","recorded_at":"study_start_date"}
    return vectors
