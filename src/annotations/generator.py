#!/usr/bin/env python

import pandas as pd

def adjust_feature_times(features_df, start_trim_sec, end_trim_sec):
    """
    Adjust the time values in features_df by adding the start_trim value.
    The end_trim_sec will also be used to ensure the final frame's time is within the trim limits.
    
    Args:
        features_df (DataFrame): The features DataFrame with time_s column.
        start_trim_sec (float): The start trim offset in seconds.
        end_trim_sec (float): The end trim offset in seconds.
    
    Returns:
        DataFrame: The features DataFrame with adjusted time values.
    """
    # Adjust time_s by adding the start_trim_sec to each time_s value
    features_df['time_s'] = features_df['time_s'] + start_trim_sec
    features_df = features_df[features_df['time_s'] <= end_trim_sec]  # Trim the features based on end_trim_sec

    return features_df

def parse_annotations(annotation_file_path):
    """
    Parses annotations from a file.
    
    Args:
        annotation_file_path (str): Path to the annotation file.
    
    Returns:
        DataFrame: Parsed annotations with columns ['start_time', 'end_time', 'label'].
    """
    annotations = []
    with open(annotation_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            start_time = float(parts[2])  # Time in seconds
            end_time = float(parts[4])  # Time in seconds
            label = parts[7]  # Label of the annotation

            # Determine person based on the first letter of the label
            if label[0] == 'C':
                person = 'Child'  # Child
            elif label[0] == 'T':
                person = 'Therapist'  # Therapist
            else:
                person = 'Unknown'  # Default to Unknown if no match

            annotations.append([start_time, end_time, label, person])
    
    annotations_df = pd.DataFrame(annotations, columns=['start_time', 'end_time', 'label', 'person'])

    return annotations_df


def map_annotations_to_features(features_df, annotations_df):
    """
    Maps annotations to feature frames based on adjusted timestamps.
    
    Args:
        features_df (DataFrame): DataFrame containing the features with 'time_s' column.
        annotations_df (DataFrame): DataFrame containing annotations with 'start_time', 'end_time', 'label', 'person'.
    
    Returns:
        DataFrame: Features DataFrame with mapped annotation labels.
    """
    # Add a new column to store annotation labels
    features_df['annotation_label'] = None

    # Iterate through each feature row
    for i, row in features_df.iterrows():
        feature_time = row['time_s']
        person_label = row['person_label']  # Get the person label from feature_df
        
        # Find annotations that overlap with the feature time
        overlapping_annotations = annotations_df[
            (annotations_df['start_time'] <= feature_time) & 
            (annotations_df['end_time'] >= feature_time)
        ]
        
        # For each overlapping annotation, assign the annotation label based on person_label
        for _, annotation in overlapping_annotations.iterrows():
            if person_label == annotation['person']:
                features_df.at[i, 'annotation_label'] = annotation['label']
                break  # Assign the first matching label and exit the loop

    return features_df

def run_annotation_alignment_pipeline(annotation_file_path, features_file_path, start_trim_sec, end_trim_sec):
    """
    Runs the full annotation alignment pipeline: parses annotations, loads features, adjusts time, and maps annotations.
    
    Args:
        annotation_file_path (str): Path to the annotation file.
        features_file_path (str): Path to the feature CSV file.
        start_trim_sec (float): Start time for trimming the video.
        end_trim_sec (float): End time for trimming the video.
    
    Returns:
        DataFrame: Features with aligned annotation labels.
    """
    # Step 1: Parse annotations from file
    annotations_df = parse_annotations(annotation_file_path)
    print(f"✓ Parsed {len(annotations_df)} annotations")
    
    # Step 2: Load features from CSV file
    features_df = pd.read_csv(features_file_path)
    print(f"✓ Loaded features from {features_file_path}: {features_df.shape}")
    
    # Step 3: Adjust feature times by adding the start_trim_sec
    features_df = adjust_feature_times(features_df, start_trim_sec, end_trim_sec)

    
    # Step 4: Map annotations to feature frames
    features_df_with_labels = map_annotations_to_features(features_df, annotations_df)
    print("✓ Mapped annotations to features")
    print(features_df_with_labels) 
    # until here - OK
    # Step 5: Delete Nan labels
    features_df_with_labels = features_df_with_labels.dropna(subset=['annotation_label'])
    
    # Drop the columns 'time_min:s.ms' and 'avg_pose_conf'
    features_df_with_labels = features_df_with_labels.drop(columns=['time_min:s.ms'])
    print(features_df_with_labels)

    
    return features_df_with_labels

# Example Usage
if __name__ == "__main__":
    project_name = "14-3-2024_#15_INDIVIDUAL_[18]"
    annotation_file_path = '/home/liubov/Bureau/new/new_annotations/14-3-2024_#15_INDIVIDUAL_(18)_WAKEE_09.10.25_BL.txt'
#     features_file_path = f'/home/liubov/Bureau/new/processed_data/processed_data_{project_name}.csv'
    features_file_path = f'/home/liubov/Bureau/new/{project_name}/PosesDir/processed_data.csv'
    start_trim_sec = 165  # Example: Start trim at 165 seconds
    end_trim_sec = 1860  # Example: End trim at 1860 seconds
    
    # Run pipeline
    features_df_with_labels = run_annotation_alignment_pipeline(
        annotation_file_path=annotation_file_path,
        features_file_path=features_file_path,
        start_trim_sec=start_trim_sec,
        end_trim_sec=end_trim_sec
    )
    
    # Optionally, save the final labeled features
    output_file_path = f'/home/liubov/Bureau/new/processed_data/{project_name}_labeled_features.csv'
    features_df_with_labels.to_csv(output_file_path, index=False)
    print(f"✓ Saved labeled features to {output_file_path}")


# /home/liubov/Bureau/new/14-3-2024_#15_INDIVIDUAL_[18]/PosesDir/processed_data.csv
