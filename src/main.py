import os

# Specify the directory path you want to work with
directory_path = 'datasets/archive_3/Original Images/Original Images/FOLDS'

# List all items (files and folders) in the specified directory
items = os.listdir(directory_path)

container = []
counter = 0
counter_sum = 0
# Iterate through the items and open folders
for item in items:
    item_path = os.path.join(directory_path, item)  # Get the full path of the item
    if os.path.isdir(item_path):  # Check if it's a directory
        train_folders = os.listdir(item_path)
        
        for folder in train_folders:
            
            if (folder == "Test") :
                train_folder_path = os.path.join(item_path,folder)            
                train = os.listdir(train_folder_path)
                
                for train_set in train:
                    train_sets_path = os.path.join(train_folder_path,train_set)
                    train_sets = os.listdir(train_sets_path)                
                    for set in train_sets:
                        counter +=1

                    counter_sum += counter
                    counter =0
                
                container.append(counter_sum)
                counter_sum=0
                
print(container)