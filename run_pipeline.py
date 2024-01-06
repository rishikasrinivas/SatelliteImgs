from steps.ingest_data import retrieve_data
from steps.model_train import train_model

import torch
import torch.nn as nn
import torchvision 
from torchvision.transforms import transforms
from torch.optim import Adam
from torchvision.models import ResNet18_Weights
import pickle


PATH_TO_DATA = "/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/Satellit/SatelliteImgs/data"



model = torchvision.models.resnet18(ResNet18_Weights)
EPOCHS = 2
LR = 0.001
OPTIMIZER = Adam(model.parameters(), lr= LR)
BATCH_SIZE = 128
LOSS_FN = nn.CrossEntropyLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#update model last layer 


train_dataloader, test_dataloader= retrieve_data(PATH_TO_DATA, batch_size=BATCH_SIZE)




train_model(model, train_dataloader, test_dataloader, EPOCHS, OPTIMIZER, LOSS_FN, DEVICE)

#save model
pickle.dump(model, open("model.sav", "wb"))