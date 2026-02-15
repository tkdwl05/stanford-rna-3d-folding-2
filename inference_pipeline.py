import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from final_solution_fixed import RNATransformer, RNADataset  # Import model definition

# Config
DATA_DIR = '.'
MODEL_PATH = 'rna_best_5_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    print("Loading datasets for inference...")
    train_seqs = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
    train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    test_seqs = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    return train_seqs, train_labels, test_seqs, sample_sub

# ==========================================
# 1. Template Matching Logic (from GPT Analysis)
# ==========================================
def build_template_db(train_seqs, train_labels):
    print("Building Template Database...")
    
    # 1. Map target_id -> sequence
    seq_map = train_seqs.set_index("target_id")["sequence"].to_dict()
    
    # 2. Map target_id -> coordinates
    # Reuse logic from final_solution_fixed or simplified here
    # Check XYZ cols
    xyz_cols = ['x','y','z'] if {'x','y','z'}.issubset(train_labels.columns) else ['x_1','y_1','z_1']
    
    # Efficient grouping
    # Extract target_id/resid
    if 'target_id' not in train_labels.columns:
        id_split = train_labels['ID'].str.rsplit('_', n=1, expand=True)
        train_labels['target_id'] = id_split[0]
        train_labels['resid'] = pd.to_numeric(id_split[1])

    # Group coords
    coords_dict = {}
    # We only need templates for sequences that might match test set
    # For speed, we process all, but in competition, maybe filter
    groups = train_labels.groupby('target_id')[xyz_cols].apply(lambda x: x.values.tolist())
    coords_dict = groups.to_dict()
    
    # Pre-compute K-mers for fast search
    def get_kmers(seq, k=3):
        return {seq[i:i+k] for i in range(len(seq)-k+1)}
    
    template_db = []
    for tid, seq in seq_map.items():
        if tid in coords_dict:
            template_db.append({
                'id': tid,
                'seq': seq,
                'coords': np.array(coords_dict[tid]),
                'kmers': get_kmers(seq)
            })
            
    print(f"Template DB size: {len(template_db)}")
    return template_db

def find_templates(query_seq, template_db, top_k=5):
    # Jaccard similarity on K-mers
    q_kmers = {query_seq[i:i+3] for i in range(len(query_seq)-2)}
    
    scores = []
    for t in template_db:
        # Quick length check? Part 2 sequences might be different lengths
        # But for template matching, usually length must match or be close
        if len(t['seq']) != len(query_seq):
            continue
            
        # Jaccard
        intersection = len(q_kmers & t['kmers'])
        union = len(q_kmers | t['kmers'])
        score = intersection / union if union > 0 else 0
        scores.append((score, t))
    
    # Sort
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# ==========================================
# 2. Deep Learning Inference
# ==========================================
def predict_dl(model, sequence):
    # Tokenize
    mapping = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
    tokenized = [mapping.get(base, 0) for base in sequence]
    
    # Pad input
    # Note: Inference needs to handle varying lengths. 
    # Current model has fixed positional encoding size (1000)
    # We should batch this properly, but for single sample:
    seq_tensor = torch.tensor([tokenized], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        preds = model(seq_tensor) # (1, 5, L, 3)
        
    return preds[0].cpu().numpy() # (5, L, 3)

# ==========================================
# 3. Hybrid Pipeline
# ==========================================
def main():
    train_seqs, train_labels, test_seqs, sample_sub = load_data()
    
    # 1. Build DB
    template_db = build_template_db(train_seqs, train_labels)
    
    # 2. Load Model
    model = RNATransformer(num_preds=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("DL Model loaded.")
        model.eval()
    except FileNotFoundError:
        print("Warning: Model weights not found. DL predictions will be random (untrained).")
    
    # 3. Inference Loop
    print("Running Inference...")
    submission_data = [] # List of dicts to update sample_sub
    
    xyz_cols = []
    for k in range(1, 6):
        xyz_cols.extend([f"x_{k}", f"y_{k}", f"z_{k}"])
        
    # Prepare submission dataframe structure if needed, but we used sample_sub logic
    # Better: Update sample_sub directly? merging is slow.
    # Let's create a dictionary mapping {target_id: [pred1, pred2, ...]}
    
    results = {}
    
    for _, row in tqdm(test_seqs.iterrows(), total=len(test_seqs)):
        tid = row['target_id']
        seq = row['sequence']
        
        # A. Template Search
        matches = find_templates(seq, template_db, top_k=5)
        
        preds_for_id = []
        
        # Strategy: Fill with templates first
        for score, t in matches:
            if score > 0.8: # Threshold for "good match"
                preds_for_id.append(t['coords'])
        
        # B. If we have fewer than 5 templates, fill with DL
        if len(preds_for_id) < 5:
            # Run DL
            dl_preds = predict_dl(model, seq) # (5, L, 3)
            
            # How many needed?
            needed = 5 - len(preds_for_id)
            
            # Take top 'needed' from DL
            # Since DL heads are arbitrary 1..5, just take them in order
            for i in range(needed):
                preds_for_id.append(dl_preds[i])
                
        # Now we have 5 predictions. Ensure shape (L, 3)
        results[tid] = preds_for_id
        
    # 4. Create Submission File
    print("Generating submission file...")
    
    # ID split for sample_sub
    sample_sub_split = sample_sub['ID'].str.rsplit('_', n=1, expand=True)
    sample_sub['_tid'] = sample_sub_split[0]
    sample_sub['_resid'] = pd.to_numeric(sample_sub_split[1])
    
    # Vectorized update is hard because IDs are rows.
    # We iterate and fill.
    # Optimization: iterate by Target ID groups
    
    # To speed up, we create a new dataframe
    # But sample_sub order must be preserved? Usually yes.
    # Use grouped iterator
    
    output_rows = []
    cols = ['ID'] + xyz_cols
    
    # This part can be slow. In a real competition, use vectorized mapping.
    # Dictionary lookup approach:
    # Key: (tid, resid) -> Row Index
    # Or just reconstruction
    
    # Fast approach:
    # 1. Melt Prediction Dict to Long Format
    # tid | resid | x_1 | y_1 ...
    
    long_data = []
    for tid, preds in results.items():
        # preds is list of 5 arrays of shape (L, 3)
        L = len(preds[0])
        for i in range(L):
            resid = i + 1
            row_dict = {'target_id': tid, 'resid': resid}
            for k in range(5):
                xyz = preds[k][i]
                row_dict[f"x_{k+1}"] = xyz[0]
                row_dict[f"y_{k+1}"] = xyz[1]
                row_dict[f"z_{k+1}"] = xyz[2]
            long_data.append(row_dict)
            
    pred_df = pd.DataFrame(long_data)
    
    # 2. Merge with sample_sub to ensure order/ids
    # sample_sub needs target_id, resid columns
    # We already created them or allow merge
    
    print("Merging with submission template...")
    # Ensure sample_sub has parsing columns
    if '_tid' not in sample_sub.columns:
         id_split = sample_sub['ID'].str.rsplit('_', n=1, expand=True)
         sample_sub['_tid'] = id_split[0]
         sample_sub['_resid'] = pd.to_numeric(id_split[1])
         
    final_sub = sample_sub[['ID', '_tid', '_resid']].merge(
        pred_df, 
        left_on=['_tid', '_resid'], 
        right_on=['target_id', 'resid'], 
        how='left'
    )
    
    # Drop helper cols
    final_sub = final_sub.drop(columns=['_tid', '_resid', 'target_id', 'resid'])
    
    # Fill NaN (if any) with 0
    final_sub = final_sub.fillna(0)
    
    final_sub.to_csv("submission_hybrid.csv", index=False)
    print("Saved submission_hybrid.csv")

if __name__ == "__main__":
    main()
