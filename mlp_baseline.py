from turtle import forward
import torch

import torch.nn.functional as F
import pandas as pd
from torch import nn
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report , accuracy_score

class Main_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(306 , 200),
            nn.Linear(200 , 100),
            nn.ReLU(),
            nn.Linear(100 , 50),
            nn.Linear(50 , 7)
        )

    def forward(self , x):
        return self.model(x)


if __name__ == "__main__":

    data_train = pd.read_csv("./data/processed/train_18_countries.csv")
    data_test = pd.read_csv("./data/processed/test_18_countries.csv")
    country = "united_states"  
    Y = data_train["cases_per_mil_cat_"+ country]
    data_train = data_train.drop('cases_per_mil_cat_' + country , axis = 1)


    
    X_train = data_train.iloc[: , :]
    Y = torch.tensor(Y)

    Y = F.one_hot(Y , 7)
    Y= Y.type(torch.DoubleTensor)
    X = torch.Tensor(data_train.values)

    model = Main_Model()
    optimizer = torch.optim.SGD(model.parameters() , lr= 0.0001)
    batch_size = 32
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(0 , 1000):
        loss_final = 0
        for j in range(0 , len(X) , batch_size):
            X_batch = X[j:j+batch_size]
            Y_batch = Y[j:j+batch_size]
            output = model.forward(X_batch)
            loss = loss_fn(output , Y_batch)
            loss_final += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("loss_final" , loss_final)

    model.eval()
    Y_test = data_test["cases_per_mil_cat_"+ country]
    X_test = data_test.drop(["cases_per_mil_cat_"+ country] , axis = 1)

    X_test = torch.Tensor(X_test.values)
    
    
    pred = model.forward(X_test)

    final_pred = torch.argmax(pred ,dim = -1)


    print("Accuracy Score" , accuracy_score(final_pred , Y_test))

    print(classification_report(final_pred , Y_test))


    torch.save(model.state_dict() , "model.pt")