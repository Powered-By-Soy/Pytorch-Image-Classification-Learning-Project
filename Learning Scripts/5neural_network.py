import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.datasets import make_circles

#make classification data
n_samples = 1000

X, y = make_circles(n_samples, noise = 0.03, random_state=42)


print(len(X),len(y))

print(f"First 5 samples of X: {X[:5]}")
print(f"First 5 samples of y: {y[:5]}")

#make data frame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0], 
                        "X2": X[ :, 1],
                        "label" : y})

print(circles.head(10))

import matplotlib.pyplot as plt
plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()


#check input output shapes
print(X.shape, y.shape)

X_sample = X[0]
y_sample = y[0]

print(f"Values for one saple of X: {X_sample} and the same for y: {y_sample}")
print(f"Shape for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

#turn data into tensors
#create train and test splits
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5], y[:5])
print(type(X), X.dtype, y.dtype)

#split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, #20% data is test
                                                    random_state = 42)

print(len(X_train), len(X_test), len(y_train), len(y_test))


#build model to classify blue and red dots
#setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#construct a model (by subclassing nn.module)
#define 2 nn.Linear() layers capable of handling the shapes of our data
#define a forward()
#instantiate an instant of model class and send to device


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers capable of handling shapes of data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) #takes 2 features, upscales to 5
        #more hidden features, more learning
        self.layer_2 = nn.Linear(in_features=5, out_features=1) #layer 2 takes from layer 1 and outputs 1 feature(same shape as y)

    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x-> layer1 -> layer2 ->output


model_0 = CircleModelV0().to(device)

print(model_0)
print(next(model_0.parameters()).device)

#can be done in the class instead
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)).to(device)

print(model_0)
#predictions
print(model_0.state_dict())
#10 weight numbers because 2*5 = 10

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, Shape: {X_test.shape}")
print(f"\n First 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 labels: \n{y_test[:10]}")

#define loss function and optimizer
#torch.nn.BCEWithLogitsLoss()
#binary cross entropy

loss_fn = nn.BCEWithLogitsLoss()
#sigmoid activation function built in

#BCELoss() requires sigmoid activation on inputs prior to calling

optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

#calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

#train and test loop
# output are logits 
#logits -> preidction probabilities -> prediction labels
#conversion done by activation function (eg simoid for binary classification and softmax for multiclass)

#view first 5 outputs of forward pass on test data
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
#use sigmoid activation function on logits
y_pred_probs = torch.sigmoid(y_logits)
print (y_pred_probs)
print(torch.round(y_pred_probs))

y_preds = torch.round(y_pred_probs)

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# y_preds.squeeze()


torch.manual_seed(42)
torch.cuda.manual_seed(42)


X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
epochs = 100
for epoch in range(epochs):
    model_0.train()

    #forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #turn logits -> pred probs -> pred label

    loss = loss_fn(y_logits,
                    y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)

        test_acc = accuracy_fn(y_true= y_test, y_pred=test_pred)


    if epoch % 10 == 0:
        print(f" Epoch: {epoch} | Loss: {loss:0.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# import requests
# from pathlib import Path

#download helper functions from Learn Pytorch Repo

from helper_functions import plot_predictions, plot_decision_boundary


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
#row column index
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

# plt.show()




#improving the model
'''
add more layers
hidden units

longer running

changing activation functions
change learning
change loss
'''



class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        #z represents logits
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x))) #faster


X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

model_1 = CircleModelV1().to(device)
print(model_1)
print(next(model_1.parameters()).device)

optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000
for epoch in range(epochs):
    model_1.train()

    #forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) #turn logits -> pred probs -> pred label

    loss = loss_fn(y_logits,
                    y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)

        test_acc = accuracy_fn(y_true= y_test, y_pred=test_pred)


    if epoch % 10 == 0:
        print(f" Epoch: {epoch} | Loss: {loss:0.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train2")
plot_decision_boundary(model_1, X_train, y_train)
#row column index
plt.subplot(1,2,2)
plt.title("Test2")
plot_decision_boundary(model_1, X_test, y_test)

plt.show()



X,y = make_circles(n_samples, noise = 0.03, random_state = 42)


plt.scatter(X[:,0], X[:,1], c=y,cmap=plt.cm.RdYlBu)
plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model with non-linear activation functions

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_3 = CircleModelV2().to(device)
print(model_3)



