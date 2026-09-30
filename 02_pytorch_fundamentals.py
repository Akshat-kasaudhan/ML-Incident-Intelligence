# %% [markdown]
# # PyTorch Fundamentals for RCA Temporal Modeling
# 
# Before building our GRU model, we need to master the PyTorch fundamentals required for 
# temporal sequence modeling. This notebook covers:
# 
# * **Phase A:** Tensors and sequence shapes `[batch, time, features]`
# * **Phase B:** Custom Dataset and DataLoader for temporal sequences
# * **Phase C:** Neural Network Fundamentals (nn.Module, layers)
# * **Phase D:** A Basic Training Loop

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
print(f"PyTorch version: {torch.__version__}")

# %% [markdown]
# ## Phase A — Tensors
# 
# For sequence modeling, our data representation needs to be `[batch, time, features]`.
# Let's explore how PyTorch tensors handle this.

# %%
# 1. Creating a tensor simulating our data format
batch_size = 8
time_steps = 24
features = 290

# Create dummy temporal data
dummy_data = torch.randn(batch_size, time_steps, features)

print(f"Tensor shape: {dummy_data.shape}")
print(f"Tensor ndim:  {dummy_data.ndim}")
print(f"Tensor dtype: {dummy_data.dtype}")

# 2. Reshaping and Views
# Suppose we want to treat all time steps independently (like our baseline model)
flattened_time = dummy_data.view(batch_size * time_steps, features)
print(f"Flattened time shape (treating windows independently): {flattened_time.shape}")

# Reshaping back
reshaped_back = flattened_time.view(batch_size, time_steps, features)
print(f"Reshaped back: {reshaped_back.shape}")
print(f"Matches original: {torch.all(dummy_data == reshaped_back).item()}")

# 3. Unsqueeze and Squeeze (useful for adding/removing dimensions)
# Let's say we have a single sample and we want to pass it to a model expecting a batch
single_sample = dummy_data[0] # Shape: [time_steps, features]
print(f"Single sample shape: {single_sample.shape}")

# Add batch dimension
batched_sample = single_sample.unsqueeze(0)
print(f"Unsqueeze (add batch dim): {batched_sample.shape}")

# Remove batch dimension
squeezed_sample = batched_sample.squeeze(0)
print(f"Squeeze (remove batch dim): {squeezed_sample.shape}")

# %% [markdown]
# ## Phase B — Dataset and DataLoader
# 
# We need a Dataset that returns:
# 1. The temporal sequence `[time, features]`
# 2. The label (root-cause service)
# 3. The true sequence length (for handling variable lengths)

# %%
class RCASequenceDataset(Dataset):
    def __init__(self, num_samples=100, max_time_steps=24, num_features=290, num_classes=5):
        """
        Simulates our temporal incident data.
        In reality, this would load windows for each incident case.
        """
        super().__init__()
        self.num_samples = num_samples
        
        self.sequences = []
        self.labels = []
        self.lengths = []
        
        for _ in range(num_samples):
            # Simulate variable length sequences (e.g., incidents that end earlier/later)
            seq_len = torch.randint(10, max_time_steps + 1, (1,)).item()
            
            # Create a sequence of shape [max_time_steps, features]
            # We pad the end with zeros to ensure consistent tensor shapes in the dataset
            seq = torch.zeros(max_time_steps, num_features)
            seq[:seq_len, :] = torch.randn(seq_len, num_features)
            
            label = torch.randint(0, num_classes, (1,)).item()
            
            self.sequences.append(seq)
            self.labels.append(label)
            self.lengths.append(seq_len)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# Create dataset and dataloader
train_dataset = RCASequenceDataset(num_samples=80)
val_dataset = RCASequenceDataset(num_samples=20)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Test the DataLoader
seqs, labels, lengths = next(iter(train_loader))
print(f"Batch sequences shape: {seqs.shape}")
print(f"Batch labels shape:    {labels.shape}")
print(f"Batch lengths shape:   {lengths.shape}")
print(f"First batch lengths:   {lengths}")

# %% [markdown]
# ## Phase C — Neural Network Fundamentals
# 
# Let's build a basic feed-forward network to understand `nn.Module`, `nn.Linear`, `nn.ReLU`, and `nn.Dropout`.
# Note: This is *not* a temporal model (GRU) yet. This model simply flattens the temporal dimension or uses a specific pooling strategy to predict.

# %%
class BasicRCAModel(nn.Module):
    def __init__(self, input_features, hidden_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths):
        # x shape: [batch, time, features]
        
        # Simple baseline approach: Mean pooling across the time dimension
        # In the future, this will be replaced by the GRU!
        
        batch_size, time_steps, features = x.shape
        
        # We should only average the actual valid time steps (ignoring padding)
        # Create a mask based on lengths
        mask = torch.arange(time_steps).expand(batch_size, time_steps) < lengths.unsqueeze(1)
        # mask shape: [batch, time]
        
        # Apply mask and compute mean
        # Zero out padding elements
        x_masked = x * mask.unsqueeze(-1).float()
        
        # Sum over time, divide by actual lengths
        x_sum = x_masked.sum(dim=1)
        x_mean = x_sum / lengths.unsqueeze(1).float()
        # x_mean shape: [batch, features]
        
        # Feed-forward network
        out = self.fc1(x_mean)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out) # Shape: [batch, num_classes]
        
        return logits

# Initialize model
model = BasicRCAModel(input_features=290, hidden_dim=64, num_classes=5)
print(model)

# Test forward pass
with torch.no_grad():
    sample_logits = model(seqs, lengths)
    print(f"Logits shape: {sample_logits.shape}")

# %% [markdown]
# ## Phase D — Training
# 
# Let's write a standard PyTorch training loop using `CrossEntropyLoss` and the `Adam` optimizer.

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

print("Starting training...")
for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_seqs, batch_labels, batch_lengths in train_loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch_seqs, batch_lengths)
        
        # Compute loss
        loss = criterion(logits, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_train_loss += loss.item() * batch_seqs.size(0)
        
        # Track accuracy
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == batch_labels).sum().item()
        total_train += batch_seqs.size(0)
        
    avg_train_loss = total_train_loss / total_train
    train_acc = correct_train / total_train
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    # Disable gradient computation for validation
    with torch.no_grad():
        for batch_seqs, batch_labels, batch_lengths in val_loader:
            logits = model(batch_seqs, batch_lengths)
            loss = criterion(logits, batch_labels)
            
            total_val_loss += loss.item() * batch_seqs.size(0)
            
            preds = torch.argmax(logits, dim=1)
            correct_val += (preds == batch_labels).sum().item()
            total_val += batch_seqs.size(0)
            
    avg_val_loss = total_val_loss / total_val
    val_acc = correct_val / total_val
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

print("Training completed!")
