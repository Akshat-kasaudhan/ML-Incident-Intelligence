import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RCATemporalDataset(Dataset):
    """
    PyTorch Dataset for Temporal Root Cause Analysis.
    
    Expects a list of tensors or a list of numpy arrays, where each item 
    represents a single incident case with shape [time, features].
    """
    def __init__(self, sequences, labels, lengths, case_ids):
        """
        Args:
            sequences (list of torch.Tensor): List where each tensor is shape [time, features]
            labels (list or torch.Tensor): The root-cause labels corresponding to each sequence.
            lengths (list or torch.Tensor): The length (number of valid time windows) for each sequence.
            case_ids (list of str): List of case_id strings for tracking and error analysis.
        """
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths
        self.case_ids = case_ids
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'length': self.lengths[idx],
            'case_id': self.case_ids[idx]
        }

def rca_collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.
    
    Padding ensures all sequences in the batch have the same temporal dimension,
    which is required to create a standard PyTorch batch tensor.
    """
    # Sort the batch in descending order by length (optional but recommended for pack_padded_sequence)
    # Even if enforce_sorted=False, it's a good practice.
    batch.sort(key=lambda x: x['length'], reverse=True)
    
    sequences = [item['sequence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    case_ids = [item['case_id'] for item in batch]
    
    # Pad sequences with zeros
    # pad_sequence handles a list of tensors of shape [time, features]
    # and returns [batch, max_time, features] if batch_first=True
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=0.0
    )
    
    return padded_sequences, labels, lengths, case_ids
