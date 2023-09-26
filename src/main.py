import os

#List the names of the entries in the dataset
print(os.listdir('archive_4'))

#Create base directory
base_dir = 'base_dir'
os.mkdir(base_dir)

#Create training directory inside of the base directory
training_dir = os.path.join(base_dir,'training_dir')
os.mkdir(training_dir)

#Create test directory inside of the base directory
val_dir = os.path.join(base_dir,'val_dir')
os.mkdir(val_dir)

akiec = os.path.join(training_dir,'akiec')
os.mkdir(akiec)

bcc = os.path.join(training_dir,'bcc')
os.mkdir(bcc)

bkl = os.path.join(training_dir,'bkl')
os.mkdir(bkl)

df = os.path.join(training_dir,'df')
os.mkdir(df)

nv = os.path.join(training_dir,'nv')
os.mkdir(nv)

mel = os.path.join(training_dir,'mel')
os.mkdir(mel)

vasc = os.path.join(training_dir,'vasc')
os.mkdir(vasc)


#Create subdirectories for each diagnostic category inside of the validation directory

akiec = os.path.join(val_dir,'akiec')
os.mkdir(akiec)

bcc = os.path.join(val_dir,'bcc')
os.mkdir(bcc)

bkl = os.path.join(val_dir,'bkl')
os.mkdir(bkl)

df = os.path.join(val_dir,'df')
os.mkdir(df)

nv = os.path.join(val_dir,'nv')
os.mkdir(nv)

mel = os.path.join(val_dir,'mel')
os.mkdir(mel)

vasc = os.path.join(val_dir,'vasc')
os.mkdir(vasc)

import pandas as pd

#Read a csv file into a DataFrame
data = pd.read_csv('archive_4/HAM10000_metadata.csv')

#Output the first 5 rows
data.head()

import sklearn
from sklearn.model_selection import train_test_split

#Divide the dataset into training and test sets
_, test_set = train_test_split(data, test_size=0.2, shuffle=True, random_state=42, stratify=data['dx'])
test_set.shape

def train_or_test(x):
    test = list(test_set['image_id'])
    if str(x) in test:
        return 'test'
    else:
        return 'train'

data['train_or_test'] = data['image_id']
data['train_or_test'] = data['train_or_test'].apply(train_or_test)
training_set = data[data['train_or_test'] == 'train']

training_set['dx'].value_counts()
test_set['dx'].value_counts()

import shutil

#Transfer images in the dataset to the related directory created before

data.set_index('image_id',inplace=True)

images_1 = os.listdir('archive_4/HAM10000_images_part_1')
images_2 = os.listdir('archive_4/HAM10000_images_part_2')

training_list = training_set['image_id']
val_list = test_set['image_id']

for image in training_list:
    filename = image + '.jpg'
    label = data.loc[image,'dx']
    if filename in images_1:
        source = os.path.join('archive_4/HAM10000_images_part_1',filename)
        direction = os.path.join(training_dir,label,filename)
        shutil.copyfile(source,direction)
    if filename in images_2:
        source = os.path.join('archive_4/HAM10000_images_part_2',filename)
        direction = os.path.join(training_dir,label,filename)
        shutil.copyfile(source,direction)
        
for image in val_list:
    filename = image + '.jpg'
    label = data.loc[image,'dx']
    if filename in images_1:
        source = os.path.join('archive_4/HAM10000_images_part_1',filename)
        direction = os.path.join(val_dir,label,filename)
        shutil.copyfile(source,direction)
    if filename in images_2:
        source = os.path.join('archive_4/HAM10000_images_part_2',filename)
        direction = os.path.join(val_dir,label,filename)
        shutil.copyfile(source,direction)
#Check

print(len(os.listdir('base_dir/training_dir/nv')))
print(len(os.listdir('base_dir/training_dir/mel')))
print(len(os.listdir('base_dir/training_dir/bkl')))
print(len(os.listdir('base_dir/training_dir/bcc')))
print(len(os.listdir('base_dir/training_dir/akiec')))
print(len(os.listdir('base_dir/training_dir/vasc')))
print(len(os.listdir('base_dir/training_dir/df')))
print('\n')
print(len(os.listdir('base_dir/val_dir/nv')))
print(len(os.listdir('base_dir/val_dir/mel')))
print(len(os.listdir('base_dir/val_dir/bkl')))
print(len(os.listdir('base_dir/val_dir/bcc')))
print(len(os.listdir('base_dir/val_dir/akiec')))
print(len(os.listdir('base_dir/val_dir/vasc')))
print(len(os.listdir('base_dir/val_dir/df')))