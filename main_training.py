from model.scunet import SCUNet
from model.drunet import UNetRes
from model.scunet_noise_map import SCUNet2
from train import train_drunet, train_scunet,train_scunet2,train_vst
from utils import *


def main(model_name, dataloader):
    # Define the model
    if model_name == 'SCUNET':    
        model = SCUNet(in_nc=1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load('/home/onyxia/work/scunet_final.pth', map_location=torch.device('cpu')))
        model = train_scunet(model, dataloader, device=device ,iterations=25000)

    
    if model_name == 'SCUNET2':    
        model = SCUNet2(in_nc=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load('/home/onyxia/work/scunet2_final.pth',weights_only=True, map_location=torch.device('cpu')))
        model = train_scunet2(model, dataloader, device=device ,iterations=200000,lr=0.0001/2)


    if model_name == 'vst':    
        model = SCUNet2(in_nc=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load('/home/onyxia/work/scunet2_iter_200000.pth',weights_only=True, map_location=torch.device(device)))
        model = train_scunet2(model, dataloader, device=device ,iterations=5000)
        model.eval()

    else:
        model = UNetRes(in_nc=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = train_drunet(model, dataloader, device=device, iterations=25000)


if __name__ == '__main__':

    #extract_datazip('waterlooED.zip',"waterlooED")


       # Load datasets (replace with your dataset paths)
    dataset_paths = [
            '/home/onyxia/work/PFEElie_Steph/BDS400/real',
            #'/home/onyxia/work/PFEElie_Steph/waterlooED',
            #'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/DIV2K',
            #'C:/Users/elieg/Documents/ENSAI_3A/PFE/Code/DPIR/datasets/Flick2K'
        ]


    dataset_paths_steph = [
        '/home/onyxia/work/PFEElie_Steph/Datasets/train/BDS400',
        '/home/onyxia/work/PFEElie_Steph/Datasets/train/waterlooED',
        '/home/onyxia/work/PFEElie_Steph/Datasets/train/DIV2K',
        '/home/onyxia/work/PFEElie_Steph/Datasets/train/Flick2K'
    ]
    image_paths = load_image_paths(dataset_paths_steph)
    patch_size = 128
    noise_level_range = [0, 50]

    batch_size=16
    #batch_size=16

    
    dataset = DenoisingDataset(image_paths, patch_size, noise_level_range)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_name = 'SCUNET2'
    main(model_name, dataloader)