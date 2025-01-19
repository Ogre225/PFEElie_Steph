import os
import torch
from PIL import Image
import numpy as np
from model.drunet import UNetRes
from model.scunet import SCUNet
from model.scunet_noise_map import SCUNet2

def load_image_paths(dataset_paths):
    image_paths = []
    for dataset_path in dataset_paths:
        for filename in os.listdir(dataset_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_paths.append(os.path.join(dataset_path, filename))
    return image_paths

def load_img(img_path):
    # Charger une image en niveaux de gris et la convertir en tensor normalisé entre 0 et 1
    img = Image.open(img_path).convert('L')  # 'L' pour grayscale
    img = np.array(img) / 255.0  # Normaliser entre 0 et 1
    return torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W) format tensor

def noise_img(img, sigma):
    noise = torch.randn_like(img) * (sigma / 255.0)  # Ajouter du bruit gaussien
    img_noisy = img + noise
    noise_map = torch.full_like(img, sigma / 255.0)  # Carte de bruit constante
    return img_noisy, noise_map

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    dataset_paths_steph = ['/home/onyxia/work/PFEElie_Steph/set12/real']
    dataset_paths = ['C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/testsets/set12']
    image_paths = load_image_paths(dataset_paths)
    sigma = 25  # Niveau de bruit

    psnr_values = []  # Liste pour stocker les PSNR des images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le modèle
    #model = UNetRes(in_nc=2,out_nc=1)
    model = SCUNet2(in_nc=2)
    model.load_state_dict(torch.load('C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/own_training/scunet2_final.pth'))
    model = model.to(device)
    model.eval()  # Mode évaluation pour le modèle

    for img_path in image_paths:
        img = load_img(img_path).to(device)
        img_noisy, noise_map = noise_img(img, sigma)
        
        with torch.no_grad():
            denoised_img = model(img_noisy, noise_map)
            #denoised_img = model(img_noisy)

        # Calcul du PSNR entre l'image originale et l'image débruitée
        psnr_value = calculate_psnr(img, denoised_img)
        psnr_values.append(psnr_value.item())  # Convertir en valeur numérique et stocker

        print(f"PSNR for {os.path.basename(img_path)}: {psnr_value:.2f} dB")

    # Calcul du PSNR moyen
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    main()

### Drunet
# bruit 15, psnr = 32,78
# bruit 25, psnr = 30,44
# bruit 50, psnr = 27,38