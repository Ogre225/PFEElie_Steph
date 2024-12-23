
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
from zipfile import ZipFile



def augmentation(x, k=0, inverse=False):
    k = k % 8
    if inverse: k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1: x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=(k//2) % 4, dims=[1,2])


# Load dataset
def load_image_paths(dataset_paths):
    image_paths = []
    for dataset_path in dataset_paths:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    return image_paths


def augmentation(x, k=0, inverse=False):
    k = k % 8
    if inverse: k = [0, 1, 6, 3, 4, 5, 2, 7][k]
    if k % 2 == 1: x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=(k//2) % 4, dims=[1,2])


# Custom Dataset class to load and apply noise to grayscale images
class DenoisingDataset(Dataset):
    def __init__(self, image_paths, patch_size, noise_level_range=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.noise_level_range = noise_level_range

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[np.random.choice(len(self.image_paths))]
        # Load image in grayscale mode
        img = Image.open(self.image_paths[idx]).convert('L')  # 'L' mode is for grayscale
        img = np.array(img, dtype=np.float32) / 255.0

        # Random crop
        h, w = img.shape
        if h<128 or w<128:

            print(h)
            print(w)

    
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

        k = np.random.randint(8)

        
        img_clean = augmentation(img_clean, k)
        img_noisy= augmentation(img_noisy, k)

        return img_noisy, img_clean, noise_level_map
    


def load_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg','bmp'))]
    images = []
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L') # Convertir en niveaux de gris
        #img = Image.open(img_path).convert('RGB').convert('L')  # Conversion en RGB puis en niveaux de gris
        images.append(img)
    return images

##Extraction

def extract_datazip(
    path_src='BSD400.zip',
    path_dest='BSD400'
):

    with ZipFile(path_src, 'r') as zip_ref:
        zip_ref.extractall()
        return True



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


def calculate_psnr_overfit(img1, img2, border=0):
    """
    Calcul du PSNR pour des tenseurs PyTorch pendant l'entraînement.
    
    img1 et img2 doivent avoir la plage de valeurs [0, 255] et être des tenseurs PyTorch.
    border est une marge facultative pour exclure des pixels du calcul.
    """
    # Vérification des dimensions
    if img1.shape != img2.shape:
        raise ValueError("Les dimensions des deux images doivent être identiques.")

    # Retirer la bordure, si spécifié
    h, w = img1.shape[-2:]  # Fonctionne pour des tenseurs avec (C, H, W) ou (H, W)
    img1 = img1[..., border:h-border, border:w-border]
    img2 = img2[..., border:h-border, border:w-border]

    # Conversion au format float64
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)

    # Calcul de la MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    # Gérer les cas particuliers où la MSE est 0 (images identiques)
    if mse == 0:
        return float("inf")

    # Calcul du PSNR
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return psnr.item()  # Retourne le PSNR en format float standard


#if __name__ == '__main__':

    #extract_datazip('/home/onyxia/work/PFEElie_Steph/WaterlooED.zip',"/home/onyxia/work/PFEElie_Steph/WaterlooED")




