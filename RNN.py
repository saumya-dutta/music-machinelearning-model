import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize, LabelEncoder
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class newRNN(nn.Module): # the RNN model
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # the layer architecture
            nn.Linear(input_size,hidden_size*3),
            nn.ReLU(),
            nn.Linear(hidden_size*3,hidden_size*7),
            nn.ReLU(),
            nn.Linear(hidden_size * 7, hidden_size * 10),
            nn.ReLU(),
            nn.Linear(hidden_size * 10, hidden_size * 7),
            nn.ReLU(),
            nn.Linear(hidden_size * 7, hidden_size * 3),
            nn.ReLU(),
            nn.Linear(hidden_size*3,output_size),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits # probability of the song belonging to each of the 8 classes

warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.") # stop warning since not applicable
data = pd.read_excel('Final Dataset.xlsx', header=0) #top 10 categories - atl hip hop and canadian hip hop -- so 8 categories
category_lines = {} # initialize dict
all_categories = [] # initialize list
all_categories = pd.Series(data.iloc[:,1].unique()) # get a list of each different genre in the dataset
for category in all_categories:
    lines = data[data.iloc[:,1] == category].iloc[:,2:] # add the numerical data for each song belonging to each type of genre in the dict
    category_lines[category] = lines
n_categories = len(pd.Series(data.iloc[:, 1].unique())) # get number of genres/classes

encoder = LabelEncoder() # for encoding classes, since currently defined as strings
category_list = list(category_lines.keys())

X = data.iloc[:,2:].to_numpy() # the X data is the feature data for each song
y = pd.DataFrame(data.iloc[:,1]).to_numpy() # the y data is the genre for each song
y = encoder.fit_transform(y) # applying the label encoder to the classes

X_tensor = torch.tensor(X,dtype=torch.float32) # convert to tensor to pass to model
y_tensor = torch.tensor(y,dtype=torch.long) # convert to tensor to pass to model

# define parameters for model
input_size = X.shape[1]
hidden_size = 11
output_size = len(np.unique(category_list)) # number of classes
learning_rate = 0.00001
num_epochs = 500000
num_splits = 5

# initialize model
model = newRNN(input_size,hidden_size,output_size)
criterion = nn.CrossEntropyLoss() # loss function - typically good for classification problems
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
accuracies = []
all_confusion_matrices = []
all_true_labels = []
all_predicted_probs = []
all_roc_auc = []

# stratified k fold cross validations since different number of songs in each class - want to maintain proportions
skf = StratifiedKFold(n_splits=num_splits,shuffle=True,random_state=15)
for fold, (train_indices, test_indices) in enumerate(skf.split(X_tensor,y_tensor)):
    X_train, X_test = X_tensor[train_indices], X_tensor[test_indices] # split into train and test
    y_train, y_test = y_tensor[train_indices], y_tensor[test_indices]

    # train
    for epoch in range(num_epochs):
        model.train() # good practice
        optimizer.zero_grad()
        outputs = model(X_train) # forward step
        loss = criterion(outputs, y_train)
        loss.backward() # get loss
        optimizer.step() # take a step

        if (epoch + 1) % 10 == 0:
            print(f'Fold [{fold + 1}/{num_splits}], Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # validate
    with torch.no_grad():
        model.eval() # good practice
        outputs = model(X_test)
        _, predicted = torch.max(outputs, dim=1)

        accuracy = (predicted == y_test).sum().item() / len(y_test) # print manual accuracy for each fold
        print(f'Fold [{fold + 1}/{num_splits}], Test Accuracy: {accuracy:.2f}')

        confusion = confusion_matrix(y_test, predicted) # get confusion matrix for each fold
        all_confusion_matrices.append(confusion) # add to total confusion

        acc = accuracy_score(y_test, predicted) # use built-in sklearn accuracy for each fold
        accuracies.append(acc) # add to total accuracy

        true_labels_binary = label_binarize(y_test, classes=list(range(len(all_categories)))) # binarize labels so can find precision
        predictions_binary = label_binarize(predicted, classes=list(range(len(all_categories))))

        fpr = dict() # false positive rate
        tpr = dict() # true positive rate
        roc_auc = dict()
        for i in range(len(all_categories)): # get roc curve for each fold
            fpr[i], tpr[i], _ = roc_curve(true_labels_binary[:, 1], predictions_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i]) # calculate area under the curve

        plt.figure()
        for i in range(len(all_categories)):
            plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
                                                 ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()


avg_confusion_matrix = sum(all_confusion_matrices) / len(all_confusion_matrices) # plot averaged confusion matrix

# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(avg_confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Average Confusion Matrix')
plt.show()

# print final result
print(f"Average accuracy: {np.mean(accuracies)*100}%")#, average precision: {np.mean(precisions)*100}%")