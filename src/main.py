
import pandas as pd

# Load your CSV file into a Pandas DataFrame
data_dir = 'archive_4/HAM10000_metadata.csv'
df = pd.read_csv(data_dir)

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df.head())

# Remove rows where 'dx' is 'bkl' or 'df'
df = df[(df['dx'] != 'bkl') & (df['dx'] != 'df')]

# Display the first few rows of the filtered DataFrame
print("\nFiltered DataFrame:")
print(df.head())

# Save the filtered DataFrame to a new CSV file
filtered_data_dir = 'filtered_data.csv'
df.to_csv(filtered_data_dir, index=False)

# Now, 'filtered_data_dir' contains the path to the new CSV file with filtered data

# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# # Create folders for data
# datasets = 'Datasets'
# os.makedirs(datasets, exist_ok=True)

# # Define class names
# class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']

# # Create training and validation folders with subfolders for each class
# for dataset_type in ['train', 'test']:
#     dataset_dir = os.path.join(datasets, dataset_type)
#     os.makedirs(dataset_dir, exist_ok=True)
    
#     for class_name in class_names:
#         class_dir = os.path.join(dataset_dir, class_name)
#         os.makedirs(class_dir, exist_ok=True)

# # Read the CSV metadata
# metadata = pd.read_csv('archive_4/HAM10000_metadata.csv')


# # Split the Dataset to test and valid 
# train_set, test_set = train_test_split(metadata, test_size=0.2, shuffle=True, random_state=42, stratify=metadata['dx'])

# metadata.set_index('image_id',inplace=True)


# images_1 = os.listdir('archive_4/HAM10000_images_part_1')
# images_2 = os.listdir('archive_4/HAM10000_images_part_2')

# training_list = train_set['image_id']
# val_list = test_set['image_id']
# # Count the number of images in each class for training and validation sets
# train_class_counts = train_set['dx'].value_counts()
# val_class_counts = test_set['dx'].value_counts()

# # Get the unique class labels
# class_labels = train_set['dx'].unique()

# # Create a bar chart to visualize the dataset split
# plt.figure(figsize=(12, 6))
# plt.bar(class_labels, train_class_counts, label='Training Set', alpha=0.7)
# plt.bar(class_labels, val_class_counts, label='Validation Set', alpha=0.7, hatch='//')
# plt.xlabel('Class Labels')
# plt.ylabel('Number of Images')
# plt.title('Dataset Split: Training vs. Validation')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Save the visualization as an image file (optional)
# plt.savefig('dataset_split_visualization.png')

# # Show the bar chart
# plt.show()

# Copy Files
# for image in training_list:
#     filename = image + '.jpg'
#     label = metadata.loc[image,'dx']    
#     if filename in images_1:
#         source = os.path.join('archive_4/HAM10000_images_part_1',filename)
#         direction = os.path.join('Datasets/train',label,filename)        
#         shutil.copyfile(source,direction)
#     if filename in images_2:
#         source = os.path.join('archive_4/HAM10000_images_part_2',filename)
#         direction = os.path.join('Datasets/train',label,filename)
#         shutil.copyfile(source,direction)
        
# for image in val_list:
#     filename = image + '.jpg'
#     label = metadata.loc[image,'dx']
#     if filename in images_1:
#         source = os.path.join('archive_4/HAM10000_images_part_1',filename)
#         direction = os.path.join('Datasets/test',label,filename)
#         shutil.copyfile(source,direction)
#     if filename in images_2:
#         source = os.path.join('archive_4/HAM10000_images_part_2',filename)
#         direction = os.path.join('Datasets/test',label,filename)
#         shutil.copyfile(source,direction)

