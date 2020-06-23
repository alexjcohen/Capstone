#%%
# Load packages
import torch
import torchvision
from torchvision import transforms
from data_loaders import TrainImageLoader, AddBlur, AddNoise, TrainImageLoaderResize
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import numpy as np
import os
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    print('Model loaded!')

    # set parameters
    subset = True
    batch_size = 8
    n_batches = 25
    blur = 2
    noise = 10
    n_epoch = 1

    if batch_size == 0:
        dataset = TrainImageLoader(filepath='Data/train2017',
                                   annots_path='Data/annotations/instances_train2017.json',
                                   device=device,
                                   transform=transforms.Compose([AddBlur(2), AddNoise(0, 10, 0), transforms.ToTensor()]))
    else:
        dataset = TrainImageLoaderResize(filepath='Data/train2017',
                                         annots_path='Data/annotations/instances_train2017.json',
                                         device=device,
                                         transform=transforms.Compose([AddBlur(blur), AddNoise(0, noise, 0), transforms.ToTensor()]),
                                         size=(224, 224))

    def collate(batch):
        img = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return [img, target]

    if subset:
        torch.manual_seed(0)
        np.random.seed(0)
        subset_indices = torch.LongTensor(batch_size*n_batches).random_(0, len(dataset))
        torch.save(subset_indices, 'Models/subset_indices' + str(batch_size*n_batches) + '.pt')
        dataset = Subset(dataset, subset_indices)

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

    print('Initiating model training:')
    for epoch in range(n_epoch):
        model.train()
        i = 0
        for img, targets in tqdm(data_loader):
        # for img, targets in data_loader:
            i += 1
            img = [img.to(device) for img in img]
            losses_dict = model(img, targets)
            losses = sum(loss for loss in losses_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # print(f'Epoch: {epoch} Batch: {i}/{len(data_loader)}, Loss: {losses}')

    print('Model trained! Saving model')
    model_name = 'model_blur' + str(blur) +\
                 '_noise' + str(noise) + \
                 '_epoch' + str(n_epoch) + \
                 '_batchsize' + str(batch_size) + \
                 '_nbatch' + str(n_batches) + \
                 '.pt'

    if not os.path.exists('Models'):
        os.mkdir('Models')

    torch.save(model.state_dict(), 'Models/' + model_name)
    print('Model saved!')
