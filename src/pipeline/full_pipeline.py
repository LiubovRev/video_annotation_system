# src/pipeline/full_pipeline.py

from src.psifx.json_processor import load_json, save_parquet
from src.models.train import train_model
from src.annotations.generator import generate_annotations

def run_pipeline(video_file: str):
    df = load_json(video_file)
    df_processed = preprocess_pose(df)
    model = train_model(df_processed)
    annotations = generate_annotations(model, df_processed)
    return annotations

if __name__ == "__main__":
    video_file = "data/raw/video1.json"
    run_pipeline(video_file)
