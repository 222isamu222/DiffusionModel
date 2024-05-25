from cnn import cnn
from exp.classifier_free_guidance import UNetCond
from exp.classifier_free_guidance import Diffuser
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Generator(Diffuser):
    def __init__(self, num_timesteps=1000, device='cpu'):
        super().__init__(num_timesteps, device=device)

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None, gamma=3.0):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        if labels is None:
            labels = torch.randint(0, 10, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels, gamma)

        return x, labels
    
#params
num_timesteps = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cols, rows = 10, 2

#prepare models and diffuser
cnn_model = cnn()
diffusion_model = UNetCond(num_labels=10)
cnn_model.load_state_dict(torch.load('cnn_model.pth'))
diffusion_model.load_state_dict(torch.load('classifier_free_guidance.pth'))
generator = Generator(num_timesteps, device=device)

cnn_model.to(device)
diffusion_model.to(device)

#get generated samples
images, labels = generator.sample(diffusion_model)
images.to(device)

#get predicted labels against images with using cnn
pred_labels = cnn_model(images)
pred_labels = pred_labels.detach().to('cpu').tolist()

#visualize
fig = plt.figure(figsize=(cols, rows))
i = 0
for r in range(rows):
    for c in range(cols):
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i].detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
        ax.set_title(f'label: {labels[i]}\n pred_label: {np.argmax(pred_labels[i])}')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        i += 1
plt.tight_layout()
plt.show()