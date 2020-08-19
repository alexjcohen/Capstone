#%%
# Load packages
import torch
import torchvision
from torchvision import transforms
from data_loaders import TrainImageLoader, AddBlur, AddNoise, TrainImageLoaderResize, Rescale
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import numpy as np
import os
import argparse
warnings.filterwarnings("ignore")

# set arguments
parser = argparse.ArgumentParser(description='Evaluation script using the pycocotools library')
parser.add_argument('--n-batches', default=125, type=int, help='Number of batches of size BATCHSIZE to use during '
                                                               'training. Use 0 to train on the entire training set. '
                                                               'Default is 125')
parser.add_argument('--batchsize', default=8, type=int, help='batch size during training')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
parser.add_argument('--blur', default=0, type=int, help='Scale of blur')
parser.add_argument('--noise', default=0, type=int, help='Scale of noise')
parser.add_argument('--train-backbone', default=False, action='store_true', help='Restrict training to feature '
                                                                                 'extractor backbone')
parser.add_argument('--train-head', default=False, action='store_true', help='Restrict training to model head')
parser.add_argument('--train-scratch', default=True, action='store_false', help='Do not use pretrained weights')

if __name__ == '__main__':

    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set parameters
    batch_size = args.batchsize
    n_batches = args.n_batches
    subset = args.n_batches > 0
    blur = args.blur
    noise = args.noise
    n_epoch = args.epochs

    print(f'Training with parameters:'
          f'\n\tepochs: {n_epoch}'
          f'\n\tbatches: {n_batches}'
          f'\n\tbatch size: {batch_size}'
          f'\n\tnoise: {noise}'
          f'\n\tblur: {blur}')

    # set model training parameters
    train_backbone = args.train_backbone
    train_head = args.train_head
    pretrained = args.train_scratch
    if pretrained:
        train_backbone = False

    # import model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    model.train()
    print('Model loaded!')
    if not pretrained:
        print('Training new network')

    # set data loaders
    if batch_size == 1:
        dataset = TrainImageLoader(filepath='Data/train2017',
                                   annots_path='Data/annotations/instances_train2017.json',
                                   device=device,
                                   transform=transforms.Compose([AddBlur(0),
                                                                 AddNoise(0, 0, 0),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                                                      (0.229, 0.224, 0.225)),
                                                                 Rescale()
                                                                 ]))
    else:
        dataset = TrainImageLoaderResize(filepath='Data/train2017',
                                         annots_path='Data/annotations/instances_train2017.json',
                                         device=device,
                                         transform=transforms.Compose([AddBlur(blur),
                                                                       AddNoise(0, noise, 0),
                                                                       transforms.ToTensor(),
                                                                       ]),
                                         size=(224, 224))

    # define the collate function for the data loader
    def collate(batch):
        img = [item[0] for item in batch]
        target = [item[1] for item in batch]
        if len(target) > 0:
            return [img, target]

    # subset the dataset if not using the full batch
    if subset:
        torch.manual_seed(42)
        np.random.seed(42)
        subset_indices = torch.LongTensor(batch_size*n_batches).random_(0, len(dataset))
        dataset = Subset(dataset, subset_indices)

    # set the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    # freeze model head if training only the CNN backbone
    # if args.train_backbone:
    if train_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.backbone.parameters():
            param.requires_grad = True
        print('Training model backbone')

    if train_head:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        for param in model.rpn.parameters():
            param.requires_grad = True
        print('Training model head')

    # get the parameters for the optimizer and move model to the device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.0005, momentum=0.9)
    model.to(device)

    # train loop
    print('Training model...')
    for epoch in range(n_epoch):

        # initialize values
        batch = 0
        total_losses = []

        # loop data loader
        for img, targets in tqdm(data_loader):

            batch += 1

            # move image tensors to device
            img = [img.to(device) for img in img]

            # convert wh to xy and rescale
            for i in range(batch_size):
                for j in range(len(targets[i]['boxes'])):

                    # convert (x,y,w,h) to (x1, y1, x2, y2)
                    targets[i]['boxes'][j][2] = targets[i]['boxes'][j][2] + targets[i]['boxes'][j][0]
                    targets[i]['boxes'][j][3] = targets[i]['boxes'][j][3] + targets[i]['boxes'][j][1]

                    # rescale image by width and height ratios
                    targets[i]['boxes'][j][0] = targets[i]['boxes'][j][0] * targets[i]['w_r']
                    targets[i]['boxes'][j][1] = targets[i]['boxes'][j][1] * targets[i]['h_r']
                    targets[i]['boxes'][j][2] = targets[i]['boxes'][j][2] * targets[i]['w_r']
                    targets[i]['boxes'][j][3] = targets[i]['boxes'][j][3] * targets[i]['h_r']

                # clean up dictionary
                del targets[i]['w_r']
                del targets[i]['h_r']

            null_check = []

            for i in range(batch_size):
                if len(targets[i]['boxes']) == 0:
                    null_check.append(i)

            null_check = null_check[::-1]

            if null_check:
                for null in null_check:
                    del targets[null]
                    del img[null]

            assert (len(img) == len(targets))

            for i in range(len(img)):
                assert len(img[i]) > 0
                assert len(targets[i]['boxes']) > 0

            # ensure boxes are smaller than the reshaped image size
            assert [target['boxes'] <= 224 for target in targets]

            # get model losses
            losses_dict = model(img, targets)
            losses = sum(loss for loss in losses_dict.values())
            assert not np.isnan(losses.item())

            if not np.isnan(losses.item()):

                # append and backprop loss
                total_losses.append(losses.item())
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # gradient clipping to prevent overflow
                optimizer.step()

            # print intermediate loss
            if batch % 500 == 0:
                print(f'Epoch: {epoch + 1} Batch: {batch}/{len(data_loader)}, '
                      f'Loss: {np.nanmean(total_losses[-500:])}')

        # print epoch average loss
        print(f'--------EPOCH: {epoch+1}, AVG. LOSS: {np.nanmean(total_losses)}--------')

    if train_backbone & train_head:
        model_end = 'full_retrain.pt'

    elif train_backbone:
        model_end = 'train_backbone.pt'

    elif train_head:
        model_end = 'train_head.pt'

    else:
        model_end = 'scratch_train.pt'

    # print model training
    print('Model trained! Saving model')
    model_name = 'model_blur' + str(blur) +\
                 '_noise' + str(noise) + \
                 '_epoch' + str(n_epoch) + \
                 '_batchsize' + str(batch_size) + '_' + \
                 str(n_batches) + '_nbatch' + \
                 model_end

    if not os.path.exists('Models'):
        os.mkdir('Models')

    # save model weights
    torch.save(model.state_dict(), 'Models/' + model_name)
    torch.save(total_losses, 'Models/total_loss' + model_name)
    print('Model saved!')
