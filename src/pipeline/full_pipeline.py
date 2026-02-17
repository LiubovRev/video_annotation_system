def run_pipeline(video_path):

    frames = psifx.process_video(video_path)

    poses = extract_pose(frames)

    predictions = model.predict(poses)

    annotations = generate_annotations(predictions)

    return annotations
