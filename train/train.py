"""
Temporal GAT Training Script for Point Cloud Sequences
"""

import numpy as np
import torch
import torch.nn as nn
import os
import random
import time
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

# Import the model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.network import DeSnowGNN


# ==================== CONFIGURATION - EDIT THESE VALUES ====================
# Input/Output Directories
DATA_FOLDER = "/root/autodl-tmp/wads_2/train_graph_data/data"           # Path to input graph data
SAVE_DIR = "/root/autodl-tmp/DeSnow-GNN/checkpoint"          # Path to save models

# Training Parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 51
STEP_SIZE = 8          # LR scheduler step size
GAMMA = 0.7            # LR scheduler decay factor
IN_CHANNELS = 4        # Input feature dimension WADS dataset is 7, CADC dataset is 4
TEMPORAL_DIM = 3       # Temporal feature dimension
RANDOM_SEED = 1      # Random seed for reproducibility

# Model Save Settings
SAVE_EPOCHS = [0, 10, 20, 30, 40, 50]  # Epochs at which to save model checkpoints

# Dataset Settings
MAX_SKIP = 1           # Maximum allowed frame skip between consecutive pairs
# ============================================================================


def set_random_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random seed
    random.seed(seed)
    # NumPy random seed
    np.random.seed(seed)
    # PyTorch random seed
    torch.manual_seed(seed)
    # Set GPU seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    """
    Custom dataset for loading consecutive graph data pairs.
    """
    def __init__(self, folder_path, max_skip=1):
        """
        Args:
            folder_path: Path to the directory containing .pt files
            max_skip: Maximum allowed frame skip between consecutive pairs
        """
        self.folder_path = folder_path
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.pt')],
            key=lambda x: int(x.split('.')[0][-5:])  # Assuming last 5 digits are numbers
        )
        self.max_skip = max_skip
        
    def __len__(self):
        return len(self.files) - 1  # Theoretical maximum length

    def __getitem__(self, idx):
        """
        Get a pair of consecutive frames.
        
        Returns:
            tuple: (current_data, prev_data) for consecutive frames
        """
        for skip in range(self.max_skip + 1):
            if idx + skip + 1 >= len(self.files):
                raise StopIteration("Reached end of dataset")
                
            current_file = self.files[idx + skip + 1]
            prev_file = self.files[idx + skip]
            
            current_num = self._extract_num(current_file)
            prev_num = self._extract_num(prev_file)
            
            if current_num == prev_num + 1:
                current_data = torch.load(os.path.join(self.folder_path, current_file))
                prev_data = torch.load(os.path.join(self.folder_path, prev_file))
                return current_data, prev_data
                
        raise ValueError(f"Could not find consecutive frames within {self.max_skip} skips")

    def _extract_num(self, filename):
        """
        Extract numeric index from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            int: Extracted numeric index
        """
        return int(filename.split('.')[0][-5:])  # Adjust based on actual format


def train(model, optimizer, criterion, data, prev, device):
    """
    Training step for a single batch.
    
    Args:
        model: The neural network model
        optimizer: Optimizer
        criterion: Loss function
        data: Current frame data
        prev: Previous frame data
        device: Device to run on
        
    Returns:
        float: Loss value
    """
    # Move data to device and prepare features
    data = data.to(device)
    prev = prev.to(device)
    
    data.x = data.x.float()
    prev.x = prev.x.float()
    
    data.x = data.x[:, 0:4]
    prev.x = prev.x[:, 0:4]
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    node_feature = model(data, data_prev=prev)
    label = data.y
    
    # Compute loss
    loss = criterion(node_feature, label)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    """Main training function."""
    
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Set random seed
    set_random_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {timestamp}")
    
    # Load dataset
    print(f"Loading dataset from {DATA_FOLDER}...")
    dataset = CustomDataset(DATA_FOLDER, max_skip=MAX_SKIP)
    loader_train = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    print(f"Dataset loaded. Total batches: {len(loader_train)}")
    
    # Initialize model
    model = DeSnowGNN(in_channels=IN_CHANNELS, temporal_dim=TEMPORAL_DIM)
    model.to(device)
    print(f"Model initialized with {IN_CHANNELS} input channels, {TEMPORAL_DIM} temporal dim")
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Training loop
    print("Starting training...")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print("-" * 60)
    
    for epoch in range(NUM_EPOCHS):
        loss_all = 0
        batch_count = 0
        
        for current, prev in loader_train:
            # Training step
            loss = train(model, optimizer, criterion, current, prev, device)
            loss_all += loss
            batch_count += 1
            
            # Clear GPU cache to prevent memory overflow
            torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = loss_all / batch_count if batch_count > 0 else 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save model at specified epochs
        if epoch in SAVE_EPOCHS:
            # Format: timestamp_seed_epoch_modelname.pth
            save_filename = f"{timestamp}_seed{RANDOM_SEED}_epoch{epoch}.pth"
            save_path = os.path.join(SAVE_DIR, save_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'seed': RANDOM_SEED,
                'timestamp': timestamp,
                'config': {
                    'data_folder': DATA_FOLDER,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'in_channels': IN_CHANNELS,
                    'temporal_dim': TEMPORAL_DIM,
                }
            }, save_path)
            print(f"  -> Model saved to {save_path}")
    
    print("-" * 60)
    print("Training completed!")
    
    # Save final model
    final_filename = f"{timestamp}_seed{RANDOM_SEED}_final.pth"
    final_path = os.path.join(SAVE_DIR, final_filename)
    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'seed': RANDOM_SEED,
        'timestamp': timestamp,
        'config': {
            'data_folder': DATA_FOLDER,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'in_channels': IN_CHANNELS,
            'temporal_dim': TEMPORAL_DIM,
        }
    }, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")