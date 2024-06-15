import os
import numpy as np
import h5py
from vif_create_feature_vec import vif_create_feature_vec
from classifier import train_classifier
from sklearn.decomposition import IncrementalPCA

def get_video_paths_and_labels(root_dir):
    paths = []
    labels = []
    
    for label, subdir in enumerate(['non_violent', 'violent']):
        subdir_path = os.path.join(root_dir, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.endswith(('.avi', '.mp4', '.mkv')):  # Adjust based on your video file formats
                paths.append(os.path.join(subdir_path, file_name))
                labels.append(label)
    
    return paths, labels

def main():
    root_dir = 'D:/Rishi/Rishi college/Year 3/sem6/MINI PROJECT/Videos to train/'  # Change this to the path containing the 'violent' and 'non_violent' directories
    
    video_paths, labels = get_video_paths_and_labels(root_dir)
    
    max_length = 0  # To track the maximum length of feature vectors
    hdf5_path = 'features.h5'

    # Extract features and determine the maximum feature vector length
    with h5py.File(hdf5_path, 'w') as hdf:
        for i, path in enumerate(video_paths):
            directory, file_name = os.path.split(path)
            try:
                feature_vec = vif_create_feature_vec(directory, file_name)
                max_length = max(max_length, len(feature_vec))
                
                # Save the feature vector and label to HDF5 without compression
                hdf.create_dataset(f'feature_{i}', data=feature_vec)
                hdf.create_dataset(f'label_{i}', data=labels[i], dtype='i')  # Store label as integer
            except Exception as e:
                print(f"Error processing {path}: {e}")
    
    # Apply Incremental PCA for dimensionality reduction
    n_components = 20  # Ensure n_components is less than or equal to the smallest batch size
    batch_size = 50  # Adjust based on available memory
    
    ipca = IncrementalPCA(n_components=n_components)
    
    with h5py.File(hdf5_path, 'r') as hdf:
        num_samples = len(video_paths)
        for i in range(0, num_samples, batch_size):
            batch_features = []
            for j in range(i, min(i + batch_size, num_samples)):
                feature = hdf[f'feature_{j}'][:]
                padded_feature = np.pad(feature, (0, max_length - len(feature)), 'constant')
                batch_features.append(padded_feature)
            batch_features = np.array(batch_features)
            ipca.partial_fit(batch_features)
    
    reduced_features = []
    final_labels = []
    
    with h5py.File(hdf5_path, 'r') as hdf:
        for i in range(num_samples):
            feature = hdf[f'feature_{i}'][:]
            padded_feature = np.pad(feature, (0, max_length - len(feature)), 'constant')
            reduced_feature = ipca.transform([padded_feature])
            reduced_features.append(reduced_feature[0])
            final_labels.append(hdf[f'label_{i}'][()])  # Retrieve label as integer
    
    reduced_features = np.vstack(reduced_features)
    final_labels = np.array(final_labels)
    
    clf = train_classifier(reduced_features, final_labels)

if __name__ == "__main__":
    main()
