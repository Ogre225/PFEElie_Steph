import os
import random
import torch
import torch.nn as nn
from utils import load_images_from_folder,DenoisingDataset,load_image_paths,calculate_psnr_overfit
from model.splines import Spline
from model.scunet_noise_map import SCUNet2
from model.scunet import SCUNet



# Training loop with dynamic learning rate adjustment and periodic weight saving
def train_scunet(model, dataloader, lr=0.0001, iterations=800000, device='cuda', save_dir="/home/onyxia/work"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
    criterion = nn.L1Loss()


    for step in range(1, iterations + 1):
        # Select a random batch

        noisy_imgs, clean_imgs, _ = next(iter(dataloader))
       
        noisy_imgs = noisy_imgs.to(torch.float32).to(device)
        clean_imgs = clean_imgs.to(torch.float32).to(device)

        # Forward pass
        output = model(noisy_imgs)
        loss = criterion(output, clean_imgs)
        psnr  = calculate_psnr_overfit(output,clean_imgs)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate every step
        scheduler.step()

        # Print loss every 500 iterations
        if step % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iteration [{step}/{iterations}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f},PSNR,{psnr}")

        if step % 5000 == 0:
            checkpoint_path = os.path.join(save_dir, f"scunet_iter_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Weights saved at iteration {step} to {checkpoint_path}")

    # Save model weights after all iterations
    final_checkpoint_path = os.path.join(save_dir, "scunet_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model weights saved at {final_checkpoint_path}")

    print("Training complete.")




# Training loop with dynamic learning rate adjustment
def train_drunet(model, dataloader, lr=0.0001, iterations=500000, device='cuda',save_dir="/home/onyxia/work"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    criterion = nn.L1Loss()

    for step in range(1, iterations + 1):
        
        noisy_imgs, clean_imgs, noise_level_map = next(iter(dataloader))
       
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        noise_level_map = noise_level_map.to(device)

        # Forward pass
        output = model(noisy_imgs, noise_level_map)
        loss = criterion(output, clean_imgs)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate every step
        scheduler.step()

        # Print loss every 500 iterations
        if step % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iteration [{step}/{iterations}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

        if step % 5000 == 0:
            checkpoint_path = os.path.join(save_dir, f"drunet_iter_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Weights saved at iteration {step} to {checkpoint_path}")

    # Save model weights after all iterations
    final_checkpoint_path = os.path.join(save_dir, "drunet_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model weights saved at {final_checkpoint_path}")

    print("Training complete.")



# Training loop with dynamic learning rate adjustment
def train_scunet2(model, dataloader, lr=0.0001, iterations=5000, device='cuda',save_dir="/home/onyxia/work"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    criterion = nn.L1Loss()

    for step in range(1, iterations + 1):
        
        noisy_imgs, clean_imgs, noise_level_map = next(iter(dataloader))
       
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        noise_level_map = noise_level_map.to(device)

        # Forward pass
        output = model(noisy_imgs, noise_level_map)
        loss = criterion(output, clean_imgs)
        psnr  = calculate_psnr_overfit(output,clean_imgs)


        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate every step
        scheduler.step()

        # Print loss every 500 iterations
        if step % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iteration [{step}/{iterations}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f},PSNR,{psnr}")

        if step % 5000 == 0:
            checkpoint_path = os.path.join(save_dir, f"scunet2_iter_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Weights saved at iteration {step} to {checkpoint_path}")

    # Save model weights after all iterations
    final_checkpoint_path = os.path.join(save_dir, "scunet2_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model weights saved at {final_checkpoint_path}")

    print("Training complete.")




def train_vst(model,dataloader, lr=0.0001, iterations=5000, device='cuda',save_dir="/home/onyxia/work"):






    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    criterion = nn.L1Loss()
    
    #model=model.load_state_dict(torch.load('/home/onyxia/work/scunet_final_25000.pth', map_location=torch.device(device)))


    spline = Spline()

    for step in range(1, iterations + 1):
        
        noisy_imgs,clean_imgs,noise_level_map = next(iter(dataloader))

        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        noise_level_map=  noise_level_map.to(device)


        trans_n = spline.forward(noisy_imgs)

        denoiser=model(trans_n,noise_level_map)
        #denoiser_bar=
        trans_inv=spline.forward(denoiser,inverse=True)

        # Forward pass
        #output = model(trans_inv)
        loss = criterion(trans_inv, clean_imgs)
        psnr  = calculate_psnr_overfit(trans_inv,clean_imgs)


        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate every step
        scheduler.step()

        # Print loss every 500 iterations
        if step % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iteration [{step}/{iterations}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f},PSNR,{psnr}")

        if step % 5000 == 0:
            checkpoint_path = os.path.join(save_dir, f"drunet_iter_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Weights saved at iteration {step} to {checkpoint_path}")

    # Save model weights after all iterations
    final_checkpoint_path = os.path.join(save_dir, "drunet_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model weights saved at {final_checkpoint_path}")

    print("Training complete.")
