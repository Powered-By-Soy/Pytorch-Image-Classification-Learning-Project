import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary



NUM_CLASSES =4
NUM_FEATURES = 2
RANDOM_SEED = 42

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

X_blob, y_blob = make_blobs(n_samples = 1000, 
                            n_features = NUM_FEATURES, 
                            centers = NUM_CLASSES, 
                            cluster_std = 1.5, 
                            random_state = RANDOM_SEED)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size = 0.2, random_state= RANDOM_SEED)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        '''
        Initializes multi-class classification model

        Args:
            input_features (int): Number of input features to the model
            output_features (int): Number of output features (number of output classes)
            hidden_units (int): Number of hidden units between layers, default 8

            Returns:

            Example:
        '''
        
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features = output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_4 = BlobModel(input_features=2, output_features= 4, hidden_units = 8).to(device)


print(model_4)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_4.parameters(), lr= 0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 100
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):

    model_4.train()
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true = y_blob_train, y_pred = y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true = y_blob_test, y_pred = test_preds)

    if epoch %10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss:{test_loss:.4f}, Test acc: {test_acc:.2f}%")






# model_4.eval()
# with torch.inference_mode():
#     y_logits = model_4(X_blob_test)

# y_pred_probs = torch.softmax(y_logits, dim=1)



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1,2,2) 
plt.title("Test")
plot_decision_boundary(model_4,X_blob_test, y_blob_test)
plt.show()



'''
Evaluation methods
accuracy
precision
recall
F1-score
Confusion matrix
Classification report
'''


from torchmetrics import Accuracy

torchmetric_accuracy = Accuracy(task = "multiclass", num_classes=NUM_CLASSES).to(device)
print(torchmetric_accuracy(test_preds, y_blob_test))


















