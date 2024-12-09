import matplotlib.pyplot as plt
import torch
import os

def plot_training_curves(checkpoint_path, save_path='training_curves.png'):
    # Load the checkpoint on cpu
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    history = checkpoint['history']
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")

checkpoint_path = "vae_best_20241209_055107.pt"
plot_training_curves(checkpoint_path)
