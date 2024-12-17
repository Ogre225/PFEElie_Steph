# Transformer la base BDS400 en niveau de gris
import os
from PIL import Image

# Chemin de la base de données d'images
input_folder = 'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/BSD1'
output_folder = 'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/BSD1_gray'

# Créer le dossier de sortie s'il n'existe pas déjà
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parcourir tous les fichiers de la base de données
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Vérifie les types d'images
        img_path = os.path.join(input_folder, filename)
        
        # Ouvrir l'image
        with Image.open(img_path) as img:
            # Convertir l'image en niveaux de gris
            gray_img = img.convert('L')
            
            # Chemin de sauvegarde de l'image transformée
            gray_img_path = os.path.join(output_folder, filename)
            
            # Sauvegarder l'image en niveaux de gris
            gray_img.save(gray_img_path)
            
        print(f"Image {filename} transformée et sauvegardée.")

print("Conversion terminée.")
