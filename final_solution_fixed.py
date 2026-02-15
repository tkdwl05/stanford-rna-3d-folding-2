import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# ==========================================
# 1. Custom Loss: Kabsch RMSD (Rotation Invariant)
# ==========================================
def kabsch_rotation(P, Q):
    """
    Finds the optimal rotation matrix R that aligns P to Q.
    P, Q: (batch, N, 3)
    Returns: Rotated P (batch, N, 3)
    """
    # Center the points
    P_mean = P.mean(dim=1, keepdim=True)
    Q_mean = Q.mean(dim=1, keepdim=True)
    P_c = P - P_mean
    Q_c = Q - Q_mean

    # Covariance matrix H
    H = torch.matmul(P_c.transpose(1, 2), Q_c) # (batch, 3, 3)

    # SVD
    # Note: torch.svd is more stable than linalg.svd for some versions, but linalg is standard
    try:
        U, S, V = torch.linalg.svd(H)
    except:
        U, S, V = torch.svd(H)

    # Correct for reflection to ensure a proper rotation (determinant must be 1)
    d = torch.det(torch.matmul(V, U.transpose(1, 2))) # (batch,)
    E = torch.eye(3, device=P.device).unsqueeze(0).repeat(P.shape[0], 1, 1)
    E[:, 2, 2] = d

    # Rotation matrix R
    R = torch.matmul(torch.matmul(V, E), U.transpose(1, 2)) # (batch, 3, 3)

    # Rotate P to align with Q
    P_aligned = torch.matmul(P_c, R.transpose(1, 2)) + Q_mean
    return P_aligned

class KabschRMSDLoss(nn.Module):
    def __init__(self):
        super(KabschRMSDLoss, self).__init__()

    def forward(self, preds, target):
        """
        preds: (batch, num_preds, seq_len, 3) - 5 predictions for Best-of-5
        target: (batch, seq_len, 3)
        Returns:
            loss: scalar (min aligned-RMSD across the 5 predictions, averaged over batch)
        """
        batch_size, num_preds, seq_len, _ = preds.shape
        
        # Calculate RMSD for each of the 5 predictions
        losses = []
        for k in range(num_preds):
            pred_k = preds[:, k, :, :] # (batch, seq_len, 3)
            
            # 1. Align pred_k to target using Kabsch
            # We must handle the case where P and Q are not perfectly alignable, Kabsch minimizes RMSD
            pred_aligned = kabsch_rotation(pred_k, target)
            
            # 2. Calculate RMSD
            mse = torch.mean((pred_aligned - target) ** 2, dim=(1, 2)) # (batch,)
            rmsd = torch.sqrt(mse + 1e-8)
            losses.append(rmsd)
        
        losses = torch.stack(losses, dim=1) # (batch, num_preds)
        
        # Best-of-5 Loss: Take the minimum loss among the 5 predictions for each sample
        # This encourages the model to generate diverse structures where at least one is close to ground truth
        min_loss, _ = torch.min(losses, dim=1) 
        
        return torch.mean(min_loss)

# ==========================================
# 2. Improved Data Loading
# ==========================================
class RNADataset(Dataset):
    def __init__(self, sequences, coordinates, max_len=200):
        self.sequences = sequences
        self.coordinates = coordinates
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.coordinates[idx]
        
        # Padding Sequence
        curr_seq_len = len(seq)
        if curr_seq_len < self.max_len:
            seq_padded = seq + [0] * (self.max_len - curr_seq_len)
        else:
            seq_padded = seq[:self.max_len]
            
        # Padding Coordinates
        coords = np.array(coords, dtype=np.float32)
        curr_coord_len = coords.shape[0]
        
        if curr_coord_len < self.max_len:
            pad_len = self.max_len - curr_coord_len
            # Pad with 0.0
            coords_padded = np.pad(coords, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        else:
            coords_padded = coords[:self.max_len]
        
        return torch.tensor(seq_padded, dtype=torch.long), torch.tensor(coords_padded, dtype=torch.float32)

def process_and_load_data(data_dir):
    print("Loading raw CSV files...")
    # Adjust filenames if necessary
    try:
        train_seq = pd.read_csv(os.path.join(data_dir, 'train_sequences.csv'))
        train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir}")
        return None, None

    # Tokenization
    print("Tokenizing sequences...")
    mapping = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
    train_seq['tokenized'] = train_seq['sequence'].apply(lambda x: [mapping.get(base, 0) for base in x])

    # ---------------------------------------------------------
    # IMPROVEMENT: Efficient Coordinate Grouping & ID Parsing
    # ---------------------------------------------------------
    print("Parsing IDs and grouping coordinates...")
    
    # ID format: target_id_residue
    # Split effectively
    id_split = train_labels['ID'].str.rsplit('_', n=1, expand=True)
    train_labels['target_id'] = id_split[0]
    train_labels['resid'] = pd.to_numeric(id_split[1])
    
    # Detect XYZ columns
    xyz_cols = ['x', 'y', 'z']
    if not set(xyz_cols).issubset(train_labels.columns):
        if set(['x_1', 'y_1', 'z_1']).issubset(train_labels.columns):
            xyz_cols = ['x_1', 'y_1', 'z_1']
        else:
            raise ValueError("Could not find x, y, z columns in labels")

    # Groupby target_id and collect coordinates in order of resid
    # This ensures (L, 3) shape aligns with sequence
    coords_dict = {}
    
    # Using a faster method than simple iteration if data is large
    # Sort by target_id and resid first
    train_labels_sorted = train_labels.sort_values(['target_id', 'resid'])
    
    # Group and aggregates lists
    # Note: allow_duplicates=False in index if needed, but here we group
    coords_grouped = train_labels_sorted.groupby('target_id')[xyz_cols].apply(lambda x: x.values.tolist())
    coords_dict = coords_grouped.to_dict()

    # Align Sequences with Coordinates
    print("Aligning sequences with coordinates...")
    final_seqs = []
    final_coords = []
    
    for _, row in tqdm(train_seq.iterrows(), total=len(train_seq)):
        tid = row['target_id']
        if tid in coords_dict:
            seq_tok = row['tokenized']
            coord = coords_dict[tid]
            
            # Length validation
            if len(seq_tok) != len(coord):
                # Simple mismatch strategy: skip or truncate?
                # For this implementation, we skip to be safe
                continue
            
            final_seqs.append(seq_tok)
            final_coords.append(coord)
            
    print(f"Processed {len(final_seqs)} valid samples.")
    return final_seqs, final_coords

# ==========================================
# 3. Model with Best-of-5 Output
# ==========================================
class RNATransformer(nn.Module):
    def __init__(self, n_tokens=5, d_model=128, nhead=8, num_layers=4, dropout=0.1, num_preds=5):
        super(RNATransformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # HEAD MODIFICATION: Output 5 separate structures
        self.num_preds = num_preds
        # Project d_model -> 3 * num_preds (x, y, z for each of 5 predictions)
        self.fc_coords = nn.Linear(d_model, 3 * num_preds) 
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        
        # Predict coordinates
        out = self.fc_coords(x) # (batch, seq_len, 3 * num_preds)
        
        # Reshape to (batch, num_preds, seq_len, 3)
        # 1. View
        out = out.view(batch_size, seq_len, self.num_preds, 3)
        # 2. Permute to get num_preds dimension second
        out = out.permute(0, 2, 1, 3)
        
        return out

# ==========================================
# 4. Main Execution Stub
# ==========================================
def main():
    # Set your data path here
    DATA_PATH = '.' # Current directory
    
    # 1. Load Data
    sequences, coordinates = process_and_load_data(DATA_PATH)
    
    if sequences is None or len(sequences) == 0:
        print("No data loaded. Exiting.")
        return

    # 2. Split Data
    train_seqs, val_seqs, train_coords, val_coords = train_test_split(sequences, coordinates, test_size=0.1, random_state=42)

    train_dataset = RNADataset(train_seqs, train_coords)
    val_dataset = RNADataset(val_seqs, val_coords)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Model
    model = RNATransformer(num_preds=5).to(device)
    print("Model Initialized with Best-of-5 Output Strategy.")
    
    # 4. Initialize Loss
    criterion = KabschRMSDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 5. Training Loop
    epochs = 5
    print(f"\nStarting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for seq, target in pbar:
            seq, target = seq.to(device), target.to(device)
            
            optimizer.zero_grad()
            preds = model(seq) # (batch, 5, seq_len, 3)
            
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)
                preds = model(seq)
                loss = criterion(preds, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Results - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save
    torch.save(model.state_dict(), 'rna_best_5_model.pth')
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()
