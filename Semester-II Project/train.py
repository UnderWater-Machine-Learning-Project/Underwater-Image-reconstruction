import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import UnderwaterDataset
from models.dna_net import DNANet

def train():
    # 1. Hardware Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training on: {device}")

    # 2. Training Hyperparameters
    BATCH_SIZE = 16  # How many images the GPU processes simultaneously
    LEARNING_RATE = 0.001 # The step size for weight updates
    EPOCHS = 50      # How many times the model sees the entire dataset

    # 3. Data Loading
    print("📂 Loading Dataset...")
    train_ds = UnderwaterDataset(
        murky_dir="datasets/Paired/trainA", 
        clear_dir="datasets/Paired/trainB"
    )
    # DataLoader handles batching and shuffling the data for better generalization
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ Found {len(train_ds)} image pairs.")

    # 4. Neural Network Setup
    model = DNANet().to(device)
    # Mean Squared Error: Calculates the pixel-perfect difference between the guess and truth
    criterion = nn.MSELoss() 
    # Adam Optimizer: Dynamically adjusts learning rates for different parameters
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. The Core Training Loop
    print("🔥 Starting Training Loop...")
    model.train() # Sets the model to training mode (enables tracking gradients)
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for i, (murky, clear) in enumerate(train_loader):
            # Move data to GPU memory
            murky = murky.to(device)
            clear = clear.to(device)

            # --- THE ALGORITHMIC CORE ---
            optimizer.zero_grad()            # 1. Clear leftover gradients from the last batch
            outputs = model(murky)           # 2. Forward Pass: The model predicts a clear image
            loss = criterion(outputs, clear) # 3. Calculate Error: Compare prediction to truth
            loss.backward()                  # 4. Backpropagation: Calculate gradients via chain rule
            optimizer.step()                 # 5. Optimize: Update network weights
            # ----------------------------

            running_loss += loss.item()
            
            # Print an update every 10 batches to monitor progress
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Current Loss: {loss.item():.4f}")
            
        # Calculate the average error for the entire epoch
        avg_loss = running_loss / len(train_loader)
        print(f"➡️ Epoch [{epoch+1}/{EPOCHS}] Completed | Average Loss: {avg_loss:.4f}\n")

        # Save the model's brain state every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"weights/dnanet_epoch_{epoch+1}.pth")

    print("🏁 Training Complete. Final weights saved to the 'weights' folder.")

if __name__ == "__main__":
    train()