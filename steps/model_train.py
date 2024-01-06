from src.model_dev import ResNet18
from steps.ingest_data import retrieve_data

def train_model(model, train_dataloader, test_dataloader, epochs, optimizer, loss_fn, device):
    res = ResNet18(device= device)
    res.develop(model)
    res.train(model, train_dataloader, test_dataloader, epochs, optimizer, loss_fn, device)
    return res