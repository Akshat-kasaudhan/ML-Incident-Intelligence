import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RCAGRUModel(nn.Module):
    """
    Temporal Root Cause Analysis model using GRU.
    
    This model processes a sequence of telemetry windows and identifies
    the likely root-cause service.
    
    Architecture:
    1. Input shape: [batch, time, features]
    2. GRU Layer(s) for temporal modeling
    3. Final valid hidden state extraction
    4. Fully Connected (Linear) layer for 5 root-cause classes
    """
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=5, dropout=0.2):
        super(RCAGRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # We only apply dropout if there is more than 1 layer in the GRU
        gru_dropout = dropout if num_layers > 1 else 0.0
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Final classification head
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        """
        Forward pass.
        
        Args:
            x (Tensor): Padded input sequences of shape [batch, max_time, features]
            lengths (Tensor): Actual length of each sequence in the batch [batch]
                              Must be on CPU and in 1D tensor format.
        
        Returns:
            Tensor: Logits for the 5 root-cause services of shape [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # Enforce lengths to be on CPU (required by pack_padded_sequence)
        lengths = lengths.cpu().to(torch.int64)
        
        # Sort sequences by length (descending) as required by pack_padded_sequence
        # Note: If enforce_sorted=False is used in pack_padded_sequence (PyTorch >= 1.1),
        # sorting manually is optional, but setting it explicitly is safer.
        # We will use enforce_sorted=False for simplicity, so no manual sorting needed.
        
        # 1. Pack the padded sequence
        # This prevents the GRU from processing the zero-padded elements 
        # and producing artificial evidence.
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # 2. Pass through GRU
        # packed_output contains all hidden states
        # hidden contains the final valid hidden state for each sequence
        # hidden shape: [num_layers, batch, hidden_size]
        packed_output, hidden = self.gru(packed_input)
        
        # 3. Extract the final hidden state from the top layer of the GRU
        # shape: [batch, hidden_size]
        final_hidden = hidden[-1, :, :]
        
        # 4. Classification
        out = self.dropout(final_hidden)
        logits = self.fc(out) # shape: [batch, num_classes]
        
        return logits

# === Example usage and tests ===
if __name__ == "__main__":
    # Simulate batch of 4 incidents, max length 20, 290 features
    batch_size = 4
    max_len = 20
    features = 290
    
    # Create random padded input
    dummy_x = torch.randn(batch_size, max_len, features)
    
    # Create variable lengths
    dummy_lengths = torch.tensor([15, 20, 5, 12])
    
    # Initialize model
    model = RCAGRUModel(input_size=features, hidden_size=64, num_layers=2)
    
    # Forward pass
    logits = model(dummy_x, dummy_lengths)
    
    print(f"Model:\n{model}\n")
    print(f"Input shape: {dummy_x.shape}")
    print(f"Lengths: {dummy_lengths}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits (raw): \n{logits}")
