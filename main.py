from model.scunet import SCUNet
from model.drunet import UNet
from train import train_drunet, train_scunet
from utils import *

def main(model_name,dataloader):
    # Define the model
    if model_name == 'SCUNET':    
        model = SCUNet(in_nc=1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = train_scunet(model,dataloader,device=device,iterations=200000)


    else:
        model = UNet()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = train_drunet(model, dataloader, device=device, iterations=200000)


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

    #batch_size=16
    #batch_size=16

    
    dataset = DenoisingDataset(image_paths, patch_size, noise_level_range)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_name = 'SCUNET'
    main(model_name,dataloader)
