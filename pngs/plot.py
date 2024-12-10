import matplotlib.pyplot as plt
import torch
import os

def plot_training_curves(checkpoint_path, save_path='rgb_training_curves.png'):
    # Load the checkpoint on cpu
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    history = checkpoint['history']
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(history['val_loss'], 'r-o', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE RGB Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")

checkpoint_path = "single_modality/vae_checkpoints/vae_RGB_best_20241209_143407.pt"
plot_training_curves(checkpoint_path)
