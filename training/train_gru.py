import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_temporal_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device="cpu"):
    """
    Standard training loop for the temporal GRU model.
    """
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_seqs, batch_labels, batch_lengths, _ in train_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            # lengths remain on CPU for pack_padded_sequence
            
            optimizer.zero_grad()
            logits = model(batch_seqs, batch_lengths)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_seqs.size(0)
            
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
        
        with torch.no_grad():
            for batch_seqs, batch_labels, batch_lengths, _ in val_loader:
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_seqs, batch_lengths)
                loss = criterion(logits, batch_labels)
                
                total_val_loss += loss.item() * batch_seqs.size(0)
                
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_labels).sum().item()
                total_val += batch_seqs.size(0)
                
        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
              
    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, best_val_loss

if __name__ == "__main__":
    print("Training loop module is ready.")
