import torch
import torch.nn as nn
import torchsummary
import torch.optim as optim

import yaml

from src.data.dataset import create_dataset
from src.utils.seeds import fix_seed
from src.visualization.visualize import plot
from src.models.models import CNN
from src.models.loss import kl_divergence
from src.models.coachs import Coach, CoachDML

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    EPOCHS = config_file['config']['epochs']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']
    METHOD = config_file['config']['method']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    model_1, model_2 = CNN(widen_factor=2).to(device), CNN(widen_factor=2).to(device)
    optimizer_1, optimizer_2 = optim.Adam(model_1.parameters(), lr=LR), optim.Adam(model_2.parameters(), lr=LR)
    loss_ce, loss_kl = nn.CrossEntropyLoss(), kl_divergence

    if METHOD == 'dml':
        coach = CoachDML([model_1, model_2], train_loader, test_loader, 
                     [loss_ce, loss_kl], [optimizer_1, optimizer_2], device, EPOCHS)
    else:
        coach = Coach(model_1, train_loader, test_loader, loss_ce, optimizer_1, device, EPOCHS)
        
    coach.train_test()

    print("accuracy: ", coach.test_acc[-1])

    plot(coach.train_loss, coach.test_loss, 'loss')
    plot(coach.train_acc, coach.test_acc, 'accuracy')

if __name__ == "__main__":
    fix_seed()
    main()