# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here
PACKAGES_TO_INSTALL = ["gdown==4.4.0", 'tensorboard']
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
# Your code here...
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.models import resnet34
import numpy as np

import subprocess
import sys

torch.manual_seed(11)
torch.cuda.manual_seed(11)
np.random.seed(11)

writer = SummaryWriter()

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    
    b_size = 256
    train_list_transforms = [transforms.RandomCrop(size=(64, 64), padding=4, padding_mode='reflect'),
                            #  transforms.ColorJitter(brightness=.5, hue=.3),
                            #  transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                             transforms.RandomHorizontalFlip(),
                            #  transforms.RandomRotation((-30, 30)),
                            #  transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    
    upd_path = os.path.join(path, kind)
    if kind == 'train':
        data_transform = transforms.Compose(train_list_transforms)
        dataset = ImageFolder(upd_path, transform=data_transform)

    elif kind == 'val':
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dataset = ImageFolder(upd_path, transform=data_transform)
        
    loader = DataLoader(dataset, batch_size=b_size,
                        shuffle=True, num_workers=0,
                        pin_memory=True)

    if kind == 'train':
        dataiter = iter(loader)
        images, _ = dataiter.next()

        img_grid = torchvision.utils.make_grid(images)
        # npimg = img_grid.numpy()
        # plt.rcParams['figure.figsize'] = (20, 30)
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))

        writer.add_image('Augmented Images', img_grid)

    return loader

class net_4_imagenet(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(net_4_imagenet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4),
            torch.nn.BatchNorm2d(384))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3456, 3456),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(3456, 3456),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(3456, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class net2_4_imagenet(torch.nn.Module):
    def __init__(self, num_classes=200):
        super(net2_4_imagenet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Dropout(),
            torch.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Dropout(),
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Dropout(),
            torch.nn.Conv2d(in_channels=128, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(384),
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(384))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3456, 3456),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(3456, 3456),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(3456, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def get_model():#loader=train_loader):
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    
    # model = net2_4_imagenet().cuda()
    # model = resnet34(pretrained=False).cuda()
    model = net_4_imagenet().cuda()
    # writer.add_graph(model)
    print('Number of weights:', np.sum([np.prod(p.shape) for p in model.parameters()]))
        
    return model
    
def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    return optimizer

# m = get_model()
# o = get_optimizer(m)
# sch = torch.optim.lr_scheduler.MultiStepLR(o, milestones=[10, 20, 30], gamma=0.1)

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
        
    batch = batch.cuda()
    logits = model.forward(batch)
    
    # criterion = torch.nn.CrossEntropyLoss()
    # predicted_classes = criterion(logits, batch)
    
    return logits

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    
    model.eval()
    iter_loader = iter(dataloader)
    loss = []
    acc = []

    for batch_image, batch_target in iter_loader:
        criterion = torch.nn.CrossEntropyLoss()
        batch_image = batch_image.cuda()
        batch_target = batch_target.cuda()

        logits = model.forward(batch_image)

        l = criterion(logits, batch_target)
        loss.append(l.item())

        n = 0   
        for y, y_hat in zip(batch_target, torch.argmax(logits, dim=1)):
            if y == y_hat:
                n += 1

        acc.append(n / len(batch_target))

    return np.mean(acc), np.mean(loss)
    
    
def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):#, scheduler=sch):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
    num_epochs = 50
    criterion = torch.nn.CrossEntropyLoss()
    best_test_acc = 0.
    flag = 0

    for epoch in tqdm(range(num_epochs)):
        flag += 1
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []

        model.train(True)
        iter_loader = iter(train_dataloader)
        
        for batch_image, batch_target in iter_loader:
            optimizer.zero_grad()
            batch_image = batch_image.cuda()
            batch_target = batch_target.cuda()
    
            logits = model.forward(batch_image)
            loss = criterion(logits, batch_target)
            train_loss.append(loss.item())

            n = 0
            for y, y_hat in zip(batch_target, torch.argmax(logits, dim=1)):
                if y == y_hat:
                    n += 1
            train_acc.append(n / len(batch_target))

            loss.backward()
            optimizer.step()

        model.eval()
        iter_loader = iter(val_dataloader)

        for batch_image, batch_target in iter_loader:
            batch_image = batch_image.cuda()
            batch_target = batch_target.cuda()

            logits = model.forward(batch_image)
            loss = criterion(logits, batch_target)
            test_loss.append(loss.item())

            n = 0   
            for y, y_hat in zip(batch_target, torch.argmax(logits, dim=1)):
                if y == y_hat:
                    n += 1
            test_acc.append(n / len(batch_target))
    
        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_acc = np.mean(train_acc)
        test_acc = np.mean(test_acc)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', test_acc, epoch)

        # subprocess.check_call(['%load_ext tensorboard'])
        # subprocess.check_call(['%tensorboard --logdir=runs'])

        # subprocess.check_call([sys.executable, "-m", "load_ext"] + ['tensorboard'])
        # subprocess.check_call([sys.executable, "-m", "tensorboard"] + ['--logdir=runs'])

        print(f'\n[Epoch {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; ' + 
            f'test loss: {test_loss:.3f}; test acc: {test_acc:.2f}')
        # scheduler.step()

        if test_acc > best_test_acc:
            flag = 0
            best_test_acc = test_acc
            # torch.save({'model_state': model.state_dict()}, 'checkpoint.pth')

        if flag == 11:
            break

    writer.close()
        
def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "a1a686ea6271ab752bb2a6aa6702b969"
    google_drive_link = "https://drive.google.com/file/d/1UGoTELM0e2Of3StC8LkBArWUUwyvKwkR/view?usp=sharing"

    return md5_checksum, google_drive_link