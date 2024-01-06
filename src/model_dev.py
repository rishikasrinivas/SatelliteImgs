import torch
import torch.nn as nn 
from torchvision.models import resnet18
from tqdm import tqdm
from src.visualizer import visualize_images
import numpy as np
from abc import ABC

import matplotlib.pyplot as plt

class Model(ABC):
     def train(self, train_data, test_data, epochs, batch_size, optimizer, loss_fn, device):
        pass
class ResNet18(Model):
    def __init__ (self, device):
        self.device = device
    
    def develop(self, model):
        model.fc = nn.Linear(in_features=512, out_features=2,bias=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), bias=False)
        print(model.parameters)
        
    def train(self, model, train_data, test_data, epochs, optimizer, loss_fn, device):
        self.develop(model)
        logging = {
            "Epochs": [],
            "Train Accuracy": [],
            "Train Loss": [],
            "Test Accuracy": [],
            "Test Loss": []
        }
        for epoch in range(1,epochs+1):
            train_acc = []
            train_loss = []
            test_acc = []
            test_loss = []
            model.train()
            for img,label in tqdm(train_data):
                optimizer.zero_grad() #0 out grads so we're starting with fresh to determine the new gradient with the new batch
                img = img.to(device)
                label = label.to(device)
         

                out = model.forward(img)
                
                loss = loss_fn(out, label)
                train_loss.append(loss.item())
                
                prediction = torch.argmax(out, axis=1)
                numCorrect = (prediction == label).sum()
                acc = numCorrect.item()/len(prediction)
                train_acc.append(acc)
                print(acc)
                #back prop
                loss.backward()
                optimizer.step()
           
            logging["Epochs"].append(epoch)
            logging["Train Accuracy"].append(np.mean(train_acc))
            logging["Train Loss"].append(np.mean(train_loss))
            print("Train Accuracy ", np.mean(train_acc))
            print("Train Loss ", np.mean(train_loss))
         
            model.eval()
    
            for img,label in tqdm(test_data):
                with torch.no_grad(): #this disables gradient calc
                    img = img.to(device)
                    label = label.to(device)
                    
                    out = model.forward(img)
                    prediction = torch.argmax(out, axis=1)
                    
                    loss = loss_fn(out, label)
                    test_loss.append(loss.item())
                    
                    numCorrect = (prediction==label).sum()
                    
                    acc= numCorrect/len(prediction)
                    test_acc.append(acc.cpu())
                    
            
            logging["Test Loss"].append(np.mean(test_loss))
            logging["Test Accuracy"].append(np.mean(test_acc))
            
            print("Test Loss ", np.mean(test_loss))
            print("Test Accuracy ", np.mean(test_acc))
        return logging
    
    def predict(self, model, image):
        pred = model(image)
        if self.device != 'cpu':
            pred = pred.detach().cpu().numpy()
        val_of_pred = np.argmax(pred, axis=1)[0]
        if val_of_pred == 0:
            return "No Wildfire"
        return "Wildfire"
        
        
        
            
                
                
                
                
                
                
    
