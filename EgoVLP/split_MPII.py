import pandas as pd

# Read the CSV file into a DataFrame
column_names = ["subject", "file_name", "start_frame", "end_frame", "activity_category_id", "activity_category_name", "split"]
df = pd.read_csv("/home/yuningc/Masking-video/EgoVLP/detectionGroundtruth-1-0.csv", names=column_names)

# Shuffle the data
df_shuffled = df.sample(frac=1, random_state=42)  # Shuffle with random seed for reproducibility

# Define the ratio for splitting (e.g., 80% for training, 20% for testing)
train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate the number of samples for each set
num_train_samples = int(len(df_shuffled) * train_ratio)
num_test_samples = len(df_shuffled) - num_train_samples

# Split the data into training and testing sets
train_data = df_shuffled[:num_train_samples]
test_data = df_shuffled[num_train_samples:]

# Write the split data to separate CSV files
train_data.to_csv("MPII_train.csv", index=False)
test_data.to_csv("MPII_test.csv", index=False)
