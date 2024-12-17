
import numpy as np
import torch
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import math



# Load dataset
def load_image_paths(dataset_paths):
    image_paths = []
    for dataset_path in dataset_paths:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    return image_paths


# Custom Dataset class to load and apply noise to grayscale images
class DenoisingDataset(Dataset):
    def __init__(self, image_paths, patch_size, noise_level_range=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.noise_level_range = noise_level_range

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image in grayscale mode
        img = Image.open(self.image_paths[idx]).convert('L')  # 'L' mode is for grayscale
        img = np.array(img, dtype=np.float32) / 255.0

        # Random crop
        h, w = img.shape
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)
        img_clean = img[top:top + self.patch_size, left:left + self.patch_size]

        # Data augmentation: random rotation by 90 degrees, flip horizontal or vertical
        #if random.random() > 0.5:
            #img_clean = np.rot90(img_clean)
        #if random.random() > 0.5:
            #img_clean = np.fliplr(img_clean)  # Flip left-right
        #if random.random() > 0.5:
            #img_clean = np.flipud(img_clean)  # Flip up-down

        # Add Gaussian noise
        noise_level = np.random.uniform(self.noise_level_range[0], self.noise_level_range[1])
        noise = np.random.normal(0, noise_level / 255.0, img_clean.shape).astype(np.float32)
        img_noisy = img_clean + noise

        # Convert to PyTorch tensors (1 channel for grayscale)
        img_clean = torch.from_numpy(img_clean).unsqueeze(0).float()  # Add channel dimension
        img_noisy = torch.from_numpy(img_noisy).unsqueeze(0).float()
        noise_level_map = torch.full((1, self.patch_size, self.patch_size), noise_level / 255.0).float()

        return img_noisy, img_clean, noise_level_map
    

# Fonction pour afficher les images
def show_images(clean_images, noisy_images, denoised_images=None):
    batch_size = clean_images.shape[0]
    fig, axes = plt.subplots(batch_size, 3 if denoised_images is not None else 2, figsize=(15, 5 * batch_size))
    
    if batch_size == 1:  # Si batch_size == 1, on ajuste l'affichage
        axes = [axes]
    
    for i in range(batch_size):
        # Afficher l'image propre
        axes[i][0].imshow(clean_images[i].permute(1, 2, 0).cpu().numpy())
        axes[i][0].set_title('Clean Image')
        axes[i][0].axis('off')
        
        # Afficher l'image bruitée
        axes[i][1].imshow(noisy_images[i].permute(1, 2, 0).cpu().numpy())
        axes[i][1].set_title('Noisy Image')
        axes[i][1].axis('off')
        
        if denoised_images is not None:
            # Afficher l'image débruitée
            axes[i][2].imshow(denoised_images[i].permute(1, 2, 0).cpu().numpy())
            axes[i][2].set_title('Denoised Image')
            axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.show()


## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == '__main__':

    # Load datasets (replace with your dataset paths)
    dataset_paths = [
            'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/BDS400'
            #'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/WaterlooB',
            #'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/DIV2K',
            #'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/Flick2K'
        ]
    image_paths = load_image_paths(dataset_paths)
    patch_size = 128
    noise_level_range = [0, 50]
    batch_size=16
    
    dataset = DenoisingDataset(image_paths, patch_size, noise_level_range)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data_iter = iter(dataloader)
    noisy_images, clean_images, sigmas = next(data_iter)

    print(f"Min: {clean_images.min()}, Max: {clean_images.max()}")
    print(f"Min: {noisy_images.min()}, Max: {noisy_images.max()}")
    print(f"Min: {sigmas.min()}, Max: {sigmas.max()}")



    # Afficher quelques images propres et bruitées
    show_images(clean_images, noisy_images)



