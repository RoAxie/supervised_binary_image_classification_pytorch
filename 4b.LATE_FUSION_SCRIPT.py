# LATE FUSION APPROACH - THREE NETS PER EVENT


"""
A list is created with three tensors, one for each plane (1 x 225 x 225).
Each plane is trained separately.
The three networks are combined together during the testing phase.
The neural networks used are: AlexNet, VGG11, and VGG19.

Saved outputs:
- average epoch loss and accuracy during training and validation
- model checkpoint at each epoch
- paths to misclassified images (at epoch 5)
"""

import numpy as np
import pandas as pd
import os
import tqdm

from scipy.sparse import load_npz

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, vgg11, vgg19

# CUSTOM DATASET

# Function to load and preprocess the image
def create_image(img_path):
    image = load_npz(img_path)  # Load sparse matrix from .npz file
    image = image.toarray()  # Convert it to a dense array
    image = np.resize(image, (1, 225, 225))

    return image

class PhysicsImageDataset(Dataset):
    def __init__(self, file_dir="path/to/files", subset='train', transform=None, target_transform=None):
        self.files = []
        self.labels = []

        self.transform = transform

        # Path to the directory containing text files with file names and labels
        txt_dir = 'path/to/txt'
        txt_file = f"{txt_dir}/{subset}_files.txt"

        with open(txt_file, 'r') as file:
            for line in file:
                fields = line.strip().split(', ')
                filenames = fields[:3]  # Get the 3 filenames for each event
                label = int(fields[-1])  # The last field is the label (0: ATMO, 1: PDK)
                # Construct the file paths for the images in 3 planes (0, 1, 2)
                file_paths = (
                    os.path.join(file_dir, "0", "ATMO" if label == 0 else "PDK", filenames[0]),
                    os.path.join(file_dir, "1", "ATMO" if label == 0 else "PDK", filenames[1]),
                    os.path.join(file_dir, "2", "ATMO" if label == 0 else "PDK", filenames[2])
                )
                self.files.append(file_paths)  # Add the file paths to the dataset
                self.labels.append(label)  # Add the label to the dataset

    
    def __len__(self):
        return len(self.files)

        
    def __getitem__(self, idx):
        # Retrieve the images and label for the given index
        img_paths = self.files[idx]
        images = []
        for img_path in img_paths:
            image = create_image(img_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        label = self.labels[idx]
        return images, label, img_paths   # Images is a list o three images

 

train_dataset = PhysicsImageDataset()
print("Training dataset created")

test_dataset = PhysicsImageDataset(subset='test')
print("Test dataset created")

#train_dataset.files[-3:]

print("Splitting the train dataset into training and validation sets")
train_set, valid_set = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
print("Data splitted")

# PARAMETERS AND FUNCTIONS

# Class prediction
prob_threshold = 0.5
method = 'mean_probability' # 'voting' or 'mean_probability'

batch_size = 128
lr = 0.0001
epochs = 10

workers = 8

# Function to handle the prediction method based on softmax probabilities or voting
def prediction(outputs, METHOD, prob_threshold = 0.5):

    if METHOD == "voting":
        _, predicted = torch.max(outputs, 1)  # Voting method: select class with the highest output score
        return predicted

    elif METHOD == "mean_probability":
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = probabilities[:, 1] >= prob_threshold  # Classify based on threshold for class 1 (PDK)
        return predicted

# Function to create a DataFrame with misclassified paths
def misclassified_paths_df(misclassified_paths, subset):
    data = []
    for file_path in misclassified_paths:
        _, filename = os.path.split(file_path)  # Extract filename from path
        base = filename.replace('.extracted.npz', '')
        base = base.split('_')

#       ['files', 'PDK', '2', 'larcv', 'plane0', '185']
        atmonu_pdk = base[1]
        mc_sim = int(base[2])
        plane = int(base[-2][-1])
        event = int(base[-1])
        
        data.append({
            'filename': filename,
            'subset' : subset,
            'atmonu_pdk': atmonu_pdk,
            'mc_sim': mc_sim,
            'plane': plane,
            'event': event
        })

    return pd.DataFrame(data)

# Specify which network to use: alexnet, vgg11, or vgg19
NETWORK = "alexnet"

def create_network():
    if NETWORK == "alexnet":
        net = alexnet(weights=None)

        # Modify the classifier to adapt to our 2-class problem
        net.classifier[4] = nn.Linear(4096,1024)
        net.classifier[6] = nn.Linear(1024,2)

        # Adjust the first conv layer to accept 1-channel input
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        
    elif NETWORK == "vgg11":
        net = vgg11(weights=None)
        #        print(net)
        net.classifier[3] = nn.Linear(4096,1024)
        net.classifier[6] = nn.Linear(1024,2)
    
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    elif NETWORK == "vgg19":
        net = vgg19(weights=None)
        #        print(net)
        net.classifier[3] = nn.Linear(4096,1024)
        net.classifier[6] = nn.Linear(1024,2)
    
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    return net

# Create data loaders for training and validation sets
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                          shuffle=False, num_workers=workers)

# PLANE 0

net0 = create_network()
net0 = net0.cuda()
net0.train()

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer0 = optim.Adam(net0.parameters(), lr=lr)

# Learning rate scheduler: reduces the learning rate after each step
scheduler = LambdaLR(optimizer0, lr_lambda=lambda step: 1 - step * 0.1)

# Dataset labels
print("Counting labels")
class0 = 0
class1 = 0
i = 0
for imgs, labels, img_paths in tqdm.tqdm(trainloader):
    class0 += (labels == 0).sum().item()
    class1 += (labels == 1).sum().item()
    i += 1
tot = class0 + class1
# Display the percentage of each class in the training set
print(f'Trainloader: AtmoNu: {class0} {100*class0/tot:.2f}%, PDK: {class1} {100*class1/tot:.2f}%')

class0 = 0
class1 = 0
i = 0
for imgs, labels, img_paths in tqdm.tqdm(validloader):
    class0 += (labels == 0).sum().item()
    class1 += (labels == 1).sum().item()
    i += 1
tot = class0 + class1
# Display the percentage of each class in the validation set
print(f'Validloader: AtmoNu: {class0} {100*class0/tot:.2f}%, PDK: {class1} {100*class1/tot:.2f}%')

# Initialization of lists for tracking metrics: loss and accuracy
plane = [0]
set_type = []
epoch_nr = []
ep_loss = []
ep_accuracy = []

# Initialization of lists to store paths of misclassified images
misclassified_train_paths = []
misclassified_valid_paths = []

# Trainloader

for epoch in tqdm.tqdm(range(epochs)):
    net0.train()

    # Initialize variables to track loss and accuracy during training
    running_loss = 0.0
    train_loss = 0.0
    running_accuracy = 0
    train_accuracy = 0
    iter_ = 0
    
    if epoch == 5:
        print('In this epoch, a list with the misclassified images will be created.')
    
    for imgs_lists,  labels, img_paths in trainloader:
        optimizer0.zero_grad()
        imgs = imgs_lists[0] # Get the first plane of the image
        imgs, labels = imgs.cuda().float(), labels.cuda()
        
        outputs = net0(imgs)
        predicted = prediction(outputs, method, prob_threshold) # Make predictions

        # Update accuracy for the current batch
        running_accuracy += torch.sum(predicted == labels) / batch_size
        train_accuracy += torch.sum(predicted == labels) / batch_size
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer0.step()
        running_loss += loss.item()
        train_loss += loss.item()

# ------------------------------------------------------------------------------------------------
        # If the epoch is 5, track misclassified images
        if epoch == 5:
            img_paths = img_paths[0]
            indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist() # Find misclassified samples
            #print('wrong',indices_wrong)
            if type(indices_wrong) is int:
                indices_wrong = [indices_wrong] # Handle the case of a single wrong index
            for idx in indices_wrong:
                misclassified_train_paths.append(img_paths[idx])
# -------------------------------------------------------------------------------------------------       
        iter_ += 1
        if iter_ % 100 == 0: # Every 100 iterations, print training progress
            accuracy = 100*running_accuracy.item()/(100)
            print(f"({epoch}:{iter_}) {running_loss/100:.3f} {accuracy:.2f}")

            # Reset running loss and accuracy after printing progress
            running_loss = 0.0
            running_accuracy = 0
    
    # Calculate average loss and accuracy for the epoch       
    avg_train_loss = train_loss/iter_
    avg_train_accuracy = 100*train_accuracy.item()/iter_
    print(f'Net0 Epoch {epoch} - Training averages: {avg_train_loss:.3f} {avg_train_accuracy:.2f}')

    # Save training metrics for the epoch
    set_type.append('train')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_train_loss, 3)) # Average epoch loss
    ep_accuracy.append(round(avg_train_accuracy, 2)) # Average epoch accuracy

    # Reset loss and accuracy accumulators for the next epoch
    avg_train_loss = 0.0
    avg_train_accuracy = 0

    # Save the model checkpoint for this epoch
    torch.save(net0, f'{NETWORK}_net0_epoch{epoch}.pth')
    print(f'{NETWORK}_net0_epoch{epoch}.pth saved')

    
    # Validloader 
    net0.eval()
    
    correct = 0
    total = 0
    valid_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculation
        for imgs_list, labels, img_paths in validloader:
            imgs = imgs_list[0] 
            imgs, labels = imgs.cuda().float(), labels.cuda()

            outputs = net0(imgs)
            predicted = prediction(outputs, method, prob_threshold)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            total += len(labels) # Total number of samples
            correct += (labels == predicted).sum().item()
            
# ---------------------------------------------------------------------------------------------------------
            # If the epoch is 5, track misclassified validation images
            if epoch == 5:
                img_paths = img_paths[0]
                indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
                #print('wrong',indices_wrong)
                if type(indices_wrong) is int:
                    indices_wrong = [indices_wrong] # Handle single index case
                for idx in indices_wrong:
                    misclassified_valid_paths.append(img_paths[idx])
# -----------------------------------------------------------------------------------------------------------

    # Calculate average validation loss and accuracy for the epoch
    #print(f"\nCorrect: Net0 - {correct}  Total: {total}")
    avg_valid_loss = valid_loss/len(validloader)
    val_accuracy = 100*correct/total
    print(f"Net0 Epoch {epoch} - Validation averages: {avg_valid_loss:.3f} {val_accuracy:.2f}\n")

    # Save validation metrics for the epoch
    set_type.append('valid')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_valid_loss, 3))
    ep_accuracy.append(round(val_accuracy, 2))

    # Update the learning rate using the scheduler
    scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}: New lr: {new_lr}')

print('Net0 training completed')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=workers)

# Counting the labels in the test set
print('Counting test labels')
class0 = 0
class1 = 0
i = 0
for imgs, labels, img_paths in tqdm.tqdm(testloader):
    class0 += (labels == 0).sum().item()
    class1 += (labels == 1).sum().item()
    i += 1
tot = class0 + class1
# Display the percentage of each class in the test set
print(f'Testloader: AtmoNu: {class0} {100*class0/tot:.2f}%, PDK: {class1} {100*class1/tot:.2f}%')

# Testloader
# Test phase: evaluation on the test set
net0.eval()

correct = 0  # Initialize counter for correct predictions
total = 0  # Initialize counter for total samples
misclassified_test_paths = []  # List to store paths of misclassified test images

# Disable gradient computation for the evaluation phase
with torch.no_grad():
    for imgs_lists, labels, img_paths in tqdm.tqdm(testloader):
        imgs = imgs_lists[0] # Use only the first plane of the event
        imgs, labels = imgs.cuda().float(), labels.cuda()
        
        outputs = net0(imgs)
        predicted = prediction(outputs, method, prob_threshold) # Get predictions

# ---------------------------------------------------------------------------------------------------------
        img_paths = img_paths[0]
        indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist() # Find misclassified samples
        #print('wrong',indices_wrong)
        if type(indices_wrong) is int:
            indices_wrong = [indices_wrong]
        for idx in indices_wrong:
            misclassified_test_paths.append(img_paths[idx])
# ----------------------------------------------------------------------------------------------------------- 

        # Update counters for accuracy calculation
        total += len(labels)
        correct += (labels == predicted).sum()

print(f"\nCorrect: {correct}, Total: {total}")
# Display the final test accuracy
print(f"Net0 - Test accuracy: {100*correct/total:.2f}\n")

print('Net0 evaluation completed')

loss_accuracy_dict0 = {
    'plane' : plane * len(set_type),
    'set' : set_type,
    'epoch' : epoch_nr,
    'loss' : ep_loss,
    'accuracy' : ep_accuracy
}
# Create a DataFrame from the metrics dictionary
df0 = pd.DataFrame(loss_accuracy_dict0)
print('Dataframe with metrics created')

# Misclassified paths

# Create DataFrames for misclassified images in the train, valid, and test sets
train_df = misclassified_paths_df(misclassified_train_paths, 'train')
valid_df = misclassified_paths_df(misclassified_valid_paths, 'valid')
test_df = misclassified_paths_df(misclassified_test_paths, 'test')

# Concatenate all misclassified images DataFrames into one
wrong_images_net0_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

# PLANE 1

net1 = create_network()

net1 = net1.cuda()
net1.train()

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters(), lr=lr)
scheduler = LambdaLR(optimizer1, lr_lambda=lambda step: 1 - step * 0.1)

plane = [1]
set_type = []
epoch_nr = []
ep_loss = []
ep_accuracy = []

misclassified_train_paths = []
misclassified_valid_paths = []

# Trainloader

for epoch in tqdm.tqdm(range(epochs)):
    net1.train()
    
    running_loss = 0.0
    train_loss = 0.0
    running_accuracy = 0
    train_accuracy = 0
    iter_ = 0

    if epoch == 5:
        print('In this epoch, a list with the misclassified images will be created.')
    
    for imgs_list, labels, img_paths in trainloader:
        optimizer1.zero_grad()
        imgs = imgs_list[1]
        imgs, labels = imgs.cuda().float(), labels.cuda() 

        outputs = net1(imgs)
        predicted = prediction(outputs, method, prob_threshold)
         
        running_accuracy += torch.sum(predicted == labels) / batch_size
        train_accuracy += torch.sum(predicted == labels) / batch_size
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        running_loss += loss.item()
        train_loss += loss.item()
        
# ------------------------------------------------------------------------------------------------
        if epoch == 5:
            img_paths = img_paths[1]
            indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
            #print('wrong',indices_wrong)
            if type(indices_wrong) is int:
                indices_wrong = [indices_wrong]
            for idx in indices_wrong:
                misclassified_train_paths.append(img_paths[idx])
# -------------------------------------------------------------------------------------------------       
        iter_ += 1
        if iter_ % 100 == 0:
            accuracy = 100*running_accuracy.item()/(100)
            print(f"({epoch}:{iter_}) {running_loss/100:.3f} {accuracy:.2f}")
            
            running_loss = 0.0
            running_accuracy = 0

                
    # Epoch training metrics: loss and accuracy        
    avg_train_loss = train_loss/iter_
    avg_train_accuracy = 100*train_accuracy.item()/iter_
    print(f'Net1 Epoch {epoch} - Training averages: {avg_train_loss:.3f} {avg_train_accuracy:.2f}')

    # Saving the metrics
    set_type.append('train')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_train_loss, 3)) # Average epoch loss
    ep_accuracy.append(round(avg_train_accuracy, 2)) # Average epoch accuracy
    
    avg_train_loss = 0.0
    avg_train_accuracy = 0

    # Saving the net
    torch.save(net1, f'{NETWORK}_net1_epoch{epoch}.pth')
    print(f'{NETWORK}_net1_epoch{epoch}.pth saved')

    
    # Validloader 
    net1.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs_list, labels, img_paths in validloader:
            imgs = imgs_list[1]
            imgs, labels = imgs.cuda().float(), labels.cuda()

            outputs = net1(imgs)
            predicted = prediction(outputs, method, prob_threshold)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            total += len(labels)
            correct += (labels == predicted).sum().item()
            
# ---------------------------------------------------------------------------------------------------------
            if epoch == 5:
                img_paths = img_paths[1]
                indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
                #print('wrong',indices_wrong)
                if type(indices_wrong) is int:
                    indices_wrong = [indices_wrong]
                for idx in indices_wrong:
                    misclassified_valid_paths.append(img_paths[idx])
# -----------------------------------------------------------------------------------------------------------    
   
    #print(f"\nCorrect: Net1 - {correct}  Total: {total}")
    avg_valid_loss = valid_loss/len(validloader)
    val_accuracy = 100*correct/total
    print(f"Net1 Epoch {epoch} - Validation averages: {avg_valid_loss:.3f} {val_accuracy:.2f}\n")

    # Saving validation metrics
    set_type.append('valid')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_valid_loss, 3))
    ep_accuracy.append(round(val_accuracy, 2))

    scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}: New lr: {new_lr}')

print('Net1 training completed')

# Testloader
net1.eval()

correct = 0
total = 0
misclassified_test_paths = []

with torch.no_grad():
    for imgs_list, labels, img_paths in tqdm.tqdm(testloader):
        imgs = imgs_list[1]
        imgs, labels = imgs.cuda().float(), labels.cuda()

        outputs = net1(imgs)
        predicted = prediction(outputs, method, prob_threshold)

        total += len(labels)
        correct += (labels == predicted).sum()
        
# ---------------------------------------------------------------------------------------------------------
        img_paths = img_paths[1]
        indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
        #print('wrong',indices_wrong)
        if type(indices_wrong) is int:
            indices_wrong = [indices_wrong]
        for idx in indices_wrong:
            misclassified_test_paths.append(img_paths[idx])
# ----------------------------------------------------------------------------------------------------------- 

print(f"\nCorrect: {correct}, Total: {total}")
print(f"Net1- Test accuracy: {100*correct/total:.2f}\n")

print('Net1 evaluation completed')

loss_accuracy_dict1 = {
    'plane' : plane * len(set_type),
    'set' : set_type,
    'epoch' : epoch_nr,
    'loss' : ep_loss,
    'accuracy' : ep_accuracy
}
df1 = pd.DataFrame(loss_accuracy_dict1)
print('Dataframe with metrics created')

train_df = misclassified_paths_df(misclassified_train_paths, 'train')
valid_df = misclassified_paths_df(misclassified_valid_paths, 'valid')
test_df = misclassified_paths_df(misclassified_test_paths, 'test')

wrong_images_net1_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

# PLANE 2

net2 = create_network()

net2 = net2.cuda()
net2.train()

criterion = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(net2.parameters(), lr=lr)
scheduler = LambdaLR(optimizer2, lr_lambda=lambda step: 1 - step * 0.1)

plane = [2]
set_type = []
epoch_nr = []
ep_loss = []
ep_accuracy = []

misclassified_train_paths = []
misclassified_valid_paths = []

# Trainloader

for epoch in tqdm.tqdm(range(epochs)):
    net2.train()
    
    running_loss = 0.0
    train_loss = 0.0
    running_accuracy = 0
    train_accuracy = 0
    iter_ = 0

    if epoch == 5:
        print('In this epoch, a list with the misclassified images will be created.')
    
    for imgs_list, labels, img_paths in trainloader:
        optimizer2.zero_grad()
        imgs = imgs_list[2]
        imgs, labels = imgs.cuda().float(), labels.cuda() 

        outputs = net2(imgs)
        predicted = prediction(outputs, method, prob_threshold)
         
        running_accuracy += torch.sum(predicted == labels) / batch_size
        train_accuracy += torch.sum(predicted == labels) / batch_size
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        train_loss += loss.item()
        
# ------------------------------------------------------------------------------------------------
        if epoch == 5:
            img_paths = img_paths[2]
            indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
            #print('wrong',indices_wrong)
            if type(indices_wrong) is int:
                indices_wrong = [indices_wrong]
            for idx in indices_wrong:
                misclassified_train_paths.append(img_paths[idx])
# -------------------------------------------------------------------------------------------------       
     
        iter_ += 1
        if iter_ % 100 == 0:
            accuracy = 100*running_accuracy.item()/(100)
            print(f"({epoch}:{iter_}) {running_loss/100:.3f} {accuracy:.2f}")
            
            running_loss = 0.0
            running_accuracy = 0
            
                
    # Epoch training metrics: loss and accuracy        
    avg_train_loss = train_loss/iter_
    avg_train_accuracy = 100*train_accuracy.item()/iter_
    print(f'Net2 Epoch {epoch} - Training averages: {avg_train_loss:.3f} {avg_train_accuracy:.2f}')

    # Saving the metrics
    set_type.append('train')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_train_loss, 3)) # Average epoch loss
    ep_accuracy.append(round(avg_train_accuracy, 2)) # Average epoch accuracy
    
    avg_train_loss = 0.0
    avg_train_accuracy = 0

    # Saving the net
    torch.save(net2, f'{NETWORK}_net2_epoch{epoch}.pth')
    print(f'{NETWORK}_net2_epoch{epoch}.pth saved')
    
    
    # Validloader 
    net2.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs_list, labels, img_paths in validloader:
            imgs = imgs_list[2]
            imgs, labels = imgs.cuda().float(), labels.cuda()

            outputs = net2(imgs)
            predicted = prediction(outputs, method, prob_threshold)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            total += len(labels)
            correct += (labels == predicted).sum().item()
# ---------------------------------------------------------------------------------------------------------
            if epoch == 5:
                img_paths = img_paths[2]
                indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
                #print('wrong',indices_wrong)
                if type(indices_wrong) is int:
                    indices_wrong = [indices_wrong]
                for idx in indices_wrong:
                    misclassified_valid_paths.append(img_paths[idx])
# -----------------------------------------------------------------------------------------------------------    
    
    #print(f"\nCorrect: Net2 - {correct}  Total: {total}")
    avg_valid_loss = valid_loss/len(validloader)
    val_accuracy = 100*correct/total
    print(f"Net2 Epoch {epoch} - Validation averages: {avg_valid_loss:.3f} {val_accuracy:.2f}\n")

    # Saving validation metrics
    set_type.append('valid')
    epoch_nr.append(epoch)
    ep_loss.append(round(avg_valid_loss, 3))
    ep_accuracy.append(round(val_accuracy, 2))

    scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}: New lr: {new_lr}')

print('Net2 training completed')

# Testloader
net2.eval()

correct = 0
total = 0
misclassified_test_paths = []

with torch.no_grad():
    for imgs_list, labels, img_paths in tqdm.tqdm(testloader):
        imgs = imgs_list[2]
        imgs, labels = imgs.cuda().float(), labels.cuda()

        outputs = net2(imgs)
        predicted = prediction(outputs, method, prob_threshold)

        total += len(labels)
        correct += (labels == predicted).sum()
        
# ----------------------------------------------------------------------------------------------------------
        img_paths = img_paths[2]
        indices_wrong = torch.nonzero(predicted != labels).squeeze().tolist()
        #print('wrong',indices_wrong)
        if type(indices_wrong) is int:
            indices_wrong = [indices_wrong]
        for idx in indices_wrong:
            misclassified_test_paths.append(img_paths[idx])
# ------------------------------------------------------------------------------------------------------------ 

print(f"\nCorrect: {correct}, Total: {total}")
print(f"Net2 - Test accuracy: {100*correct/total:.2f}\n")

print('Net2 evaluation completed')

loss_accuracy_dict2 = {
    'plane' : plane * len(set_type),
    'set' : set_type,
    'epoch' : epoch_nr,
    'loss' : ep_loss,
    'accuracy' : ep_accuracy
}
df2 = pd.DataFrame(loss_accuracy_dict2)
print('Dataframe with metrics created')

# Combining the DataFrames from all planes (df0, df1, df2) into a single DataFrame
loss_accuracy_df = pd.concat([df0, df1, df2], ignore_index=True)
#print(loss_accuracy_df)

# Save the combined DataFrame to a CSV file
loss_accuracy_df.to_csv(f'{NETWORK}_running_loss_accuracy.csv', index=False)
print('Metrics saved in a csv file')

# Misclassified
train_df = misclassified_paths_df(misclassified_train_paths, 'train')
valid_df = misclassified_paths_df(misclassified_valid_paths, 'valid')
test_df = misclassified_paths_df(misclassified_test_paths, 'test')

wrong_images_net2_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

# Combine misclassified images DataFrames from all planes (0, 1, 2)
wrong_images_df = pd.concat([wrong_images_net0_df, wrong_images_net1_df, wrong_images_net2_df], ignore_index=True)

# Save the DataFrame with misclassified image paths to a CSV file
wrong_images_df.to_csv(f'{NETWORK}_wrong_images.csv', index=False)
print('Wrong images saved in a csv file')

print(f'{NETWORK} - Running finished')