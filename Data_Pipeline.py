import numpy as np
#Assigning file paths to variables(Uses raw strings to ignore any escape sequences in the file paths)
train_val_images = r'C:\Users\sudha\OneDrive\Desktop\Python Projects\VS code\Neural Networks\archive\train_images.npy' 
train_val_labels = r'C:\Users\sudha\OneDrive\Desktop\Python Projects\VS code\Neural Networks\archive\train_labels.npy'
test_images = r'C:\Users\sudha\OneDrive\Desktop\Python Projects\VS code\Neural Networks\archive\test_images.npy'
test_labels = r'C:\Users\sudha\OneDrive\Desktop\Python Projects\VS code\Neural Networks\archive\test_labels.npy'
#Transforms npy files as NumPy arrays and loads them into the respective variables
train_val_images = np.load(train_val_images)
train_val_labels = np.load(train_val_labels)
# 80% of the training data for training, 20% for validation. By multiplying tuples first element by 0.8
train_images = train_val_images[:int(train_val_images.shape[0] * 0.8)]#Slicing the image dataset to get the first 80% of the images for training
train_labels = train_val_labels[:int(train_val_labels.shape[0] * 0.8)]
val_images = train_val_images[int(train_val_images.shape[0] * 0.8):]#Slicing the image dataset to get the last 20% of the images for validation
val_labels = train_val_labels[int(train_val_labels.shape[0] * 0.8):]
test_images = np.load(test_images)
test_labels = np.load(test_labels)
def flatten(train_images,val_images,test_images):
    train_images_f = train_images.reshape(-1,784)
    val_images_f = val_images.reshape(-1,784)
    test_images_f = test_images.reshape(-1,784)
    return train_images_f,val_images_f,test_images_f
train_images_f,val_images_f,test_images_f = flatten(train_images,val_images,test_images)
def normalize(train_images_f,val_images_f,test_images_f):
    train_images_n = train_images_f.astype(np.float32)/255.0
    val_images_n = val_images_f.astype(np.float32)/255.0
    test_images_n = test_images_f.astype(np.float32)/255.0
    return train_images_n,val_images_n,test_images_n
train_images_n,val_images_n,test_images_n = normalize(train_images_f,val_images_f,test_images_f)
X = train_images_n
Batch_Size = X.shape[0]
