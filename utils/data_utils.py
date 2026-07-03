# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np


def _get_default_device():
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MinMaxNorm01:
    """Scale data to range [0, 1]"""
    
    def __init__(self):
        pass
    
    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
    
    def transform(self, x):
        diff = self.max - self.min
        # Handle case where max == min (constant feature)
        if diff == 0:
            return np.zeros_like(x)
        x = 1.0 * (x - self.min) / diff
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True, device=None):
    """Create PyTorch DataLoader from tensors"""
    device = torch.device(device) if device is not None else _get_default_device()
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


def prepare_data(csv_file, window=5, predict=1, test_ratio=0.15, val_ratio=0.05, device=None):
    """
    Prepare data for training from CSV file
    
    Args:
        csv_file: Path to CSV file
        window: Input sequence length
        predict: Output sequence length
        test_ratio: Ratio of data for testing
        val_ratio: Ratio of data for validation
        device: Optional torch device for created tensors
    
    Returns:
        train_loader, val_loader, test_loader, mmn (normalizer)
    """
    # Load data
    X = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
    
    # Basic preprocessing
    X = X.drop(columns=["Name"])
    X["Target"] = (X["Price"].pct_change().shift(-1) > 0).astype(int)
    X.dropna(inplace=True)
    
    # Convert to numpy
    a = X.to_numpy()
    
    # Calculate split sizes (based on number of sequences)
    ran = a.shape[0]
    n_seq = ran - window
    test_len = int(test_ratio * n_seq)
    val_len = int(val_ratio * n_seq)
    train_len = n_seq - test_len - val_len
    
    # Split raw data first: determine split points in raw data
    # Test: first test_len sequences -> raw data points [0 : test_len + window]
    # Train: next train_len sequences -> raw data points [test_len : test_len + train_len + window]
    # Val: last val_len sequences -> raw data points [test_len + train_len : end]
    
    test_end_idx = test_len + window
    train_start_idx = test_len
    train_end_idx = test_len + train_len + window
    val_start_idx = test_len + train_len
    
    # Split raw data
    a_test = a[:test_end_idx]
    a_train = a[train_start_idx:train_end_idx]
    a_val = a[val_start_idx:]
    
    # Fit normalizer on training data only
    mmn = MinMaxNorm01()
    mmn.fit(a_train)
    
    # Transform each split separately using the same normalizer
    a_test_norm = mmn.transform(a_test)
    a_train_norm = mmn.transform(a_train)
    a_val_norm = mmn.transform(a_val)
    
    # Create sequences from each normalized split
    def create_sequences(data, start_offset=0):
        """Create sequences from normalized data"""
        X_seq = []
        Y_seq = []
        data_len = data.shape[0]
        i = start_offset
        while i + window < data_len:
            X_seq.append(torch.Tensor(data[i:i+window, 1:]))
            Y_seq.append(torch.Tensor(data[i+window:i+window+predict, 0]))
            i += 1
        if len(X_seq) > 0:
            XX = torch.stack(X_seq, dim=0)
            YY = torch.stack(Y_seq, dim=0)
            YY = YY[:, :, None]
            return XX, YY
        return None, None
    
    # Create sequences for each split
    X_test, Y_test = create_sequences(a_test_norm, start_offset=0)
    X_train, Y_train = create_sequences(a_train_norm, start_offset=0)
    X_val, Y_val = create_sequences(a_val_norm, start_offset=0)
    
    # Create tensors on the selected runtime device
    device = torch.device(device) if device is not None else _get_default_device()
    X_test = X_test.float().to(device) if X_test is not None else None
    Y_test = Y_test.float().to(device) if Y_test is not None else None

    X_train = X_train.float().to(device) if X_train is not None else None
    Y_train = Y_train.float().to(device) if Y_train is not None else None

    X_val = X_val.float().to(device) if X_val is not None else None
    Y_val = Y_val.float().to(device) if Y_val is not None else None
    
    # Get number of features (from X_train which should always exist)
    num_features = X_train.shape[2] if X_train is not None else a.shape[1] - 1
    
    # Create data loaders
    train_loader = data_loader(X_train, Y_train, 64, shuffle=False, drop_last=False, device=device)
    val_loader = data_loader(X_val, Y_val, 64, shuffle=False, drop_last=False, device=device)
    test_loader = data_loader(X_test, Y_test, 64, shuffle=False, drop_last=False, device=device)
    
    return train_loader, val_loader, test_loader, mmn, num_features
