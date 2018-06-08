import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256 #784
image_size = 2352 #2352
num_epochs = 200
batch_size = 32
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])

# MNIST dataset
#mnist = torchvision.datasets.MNIST(root='../../data/',
#                                    train=True,
#                                    transform=transform,
#                                    download=True)

wikiart = torchvision.datasets.ImageFolder(root='./images_abstract',
                                           transform=transform)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=wikiart,
                                          batch_size=batch_size, 
                                          shuffle=True)

nc = 3
ndf = 28
# Discriminator
#D = nn.Sequential(
#    nn.Linear(image_size, hidden_size),
#    nn.LeakyReLU(0.2),
#    nn.BatchNorm1d(hidden_size),
#    nn.Linear(hidden_size, hidden_size),
#    nn.LeakyReLU(0.2),
#    nn.BatchNorm1d(hidden_size),
#    nn.Linear(hidden_size, 1),
#    nn.Sigmoid())
D = nn.Sequential(
    nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias = False),
    nn.LeakyReLU(0.2),
    
    nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias = False),
    nn.BatchNorm1d(ndf*2),
    nn.LeakyReLU(0.2),
    
    
    nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias = False),
    nn.BatchNorm1d(ndf*4),
    nn.LeakyReLU(0.2),
    
    nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=1, bias = False),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(ndf*8),

    nn.Conv2d(ndf*8, 1, 4, stride=2, padding=0, bias = False),
    nn.Sigmoid())

# Generator 
# G = nn.Sequential(
#     nn.Linear(latent_size, hidden_size),
#     nn.ReLU(),
#     nn.Linear(hidden_size, hidden_size),
#     nn.ReLU(),
#     nn.Linear(hidden_size, image_size),
#     nn.Tanh())
nz = 100
ngf = 28
G = nn.Sequential(
    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8),
    nn.ReLU(True),
    # state size. (ngf*8) x 4 x 4
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(True),
    # state size. (ngf*4) x 8 x 8
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(True),
    # state size. (ngf*2) x 16 x 16
    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),
    # state size. (ngf) x 32 x 32
    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4)
g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        if i == 468: 
            break
        #images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        print(images.shape)
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 3, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled image
    fake_images = fake_images.reshape(fake_images.size(0), 3, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')