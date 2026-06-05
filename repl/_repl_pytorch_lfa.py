"""
repl/_repl_pytorch_lfa.py
PyTorch binary classifier for lateral flow assay (LFA) COVID-like test.
Synthetic data generation, data augmentation, CLI argparse pattern,
confidence-threshold decision: positive -> send person home.
Connects to lab-on-chip evanescent sensor (refractive index signals).
Edge deployment: ONNX export for MCU/RPi inference.
"""
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 0. ARGPARSE: ALL OPTIONS  (the "read all options" ask)
# ============================================================
def make_parser():
    p = argparse.ArgumentParser(
        description="LFA classifier: train or infer COVID-like test result",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # data
    p.add_argument("--n-samples",  type=int,   default=2000,   help="synthetic training samples")
    p.add_argument("--pos-frac",   type=float, default=0.30,   help="fraction positive (prevalence)")
    p.add_argument("--noise-std",  type=float, default=0.05,   help="sensor noise std dev")
    p.add_argument("--seed",       type=int,   default=42,     help="RNG seed")
    # model
    p.add_argument("--hidden",     type=int,   default=32,     help="hidden layer width")
    p.add_argument("--dropout",    type=float, default=0.20,   help="dropout rate (augmentation)")
    # training
    p.add_argument("--epochs",     type=int,   default=40,     help="training epochs")
    p.add_argument("--lr",         type=float, default=1e-3,   help="learning rate")
    p.add_argument("--batch-size", type=int,   default=64,     help="mini-batch size")
    p.add_argument("--val-frac",   type=float, default=0.20,   help="validation split fraction")
    # decision
    p.add_argument("--threshold",  type=float, default=0.50,   help="positive decision threshold")
    p.add_argument("--mode",       choices=["train","infer","demo"], default="demo")
    # deployment
    p.add_argument("--export-onnx", action="store_true",       help="export ONNX for MCU/RPi")
    return p

# Parse with defaults (no sys.argv in REPL)
args = make_parser().parse_args([])

print("=" * 62)
print("PYTORCH LFA CLASSIFIER  (COVID-like lateral flow assay)")
print("=" * 62)
print()
print("=== 0. CLI OPTIONS (argparse) ===")
print("  All options with defaults:")
for k, v in vars(args).items():
    print(f"    --{k.replace('_','-'):20s}  {str(v):10s}")
print()
print("  Example usage:")
print("    py -3.12 _repl_pytorch_lfa.py --mode train --epochs 80 --threshold 0.60")
print("    py -3.12 _repl_pytorch_lfa.py --mode infer --export-onnx")
print()

# ============================================================
# 1. SYNTHETIC LFA DATA GENERATION
# ============================================================
print("=== 1. SYNTHETIC DATA: LATERAL FLOW ASSAY ===")
print("""
  Lateral flow assay (LFA) features (what fiber sensor measures):
    T_line   : test line intensity    (antibody-antigen binding)
    C_line   : control line intensity (flow confirmation)
    BG       : background intensity
    ratio    : T_line / C_line        (key diagnostic ratio)
    flow_time: seconds to reach read zone
    T_C_diff : T_line - C_line

  POSITIVE test:  T_line HIGH (antigen captured), ratio > 1
  NEGATIVE test:  T_line LOW  (no antigen),       ratio < 0.3
  INVALID test:   C_line LOW  (bad flow)           ratio undefined

  Signal model (from Beer-Lambert + evanescent sensing):
    I = I0 * exp(-eps * c_antigen * L_eff)
    -> captured on fiber evanescent sensor or CCD strip reader
""")

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

N     = args.n_samples
fpos  = args.pos_frac
noise = args.noise_std

def make_lfa_data(N, fpos, noise_std, rng):
    n_pos = int(N * fpos)
    n_neg = N - n_pos
    labels = np.array([1]*n_pos + [0]*n_neg, dtype=np.float32)

    # Feature generation (physics-based)
    T_line = np.zeros(N, dtype=np.float32)
    C_line = np.zeros(N, dtype=np.float32)
    BG     = np.zeros(N, dtype=np.float32)
    flow_t = np.zeros(N, dtype=np.float32)

    # Positive: T_line high (antigen captured)
    T_line[:n_pos] = rng.normal(0.75, 0.10, n_pos)  # strong test line
    T_line[n_pos:] = rng.normal(0.12, 0.05, n_neg)  # weak / absent

    # Control line: always present if flow is good
    C_line = rng.normal(0.65, 0.08, N)

    # Background: Poisson-like noise from non-specific binding
    BG = rng.exponential(0.04, N).astype(np.float32)

    # Flow time: slower for positive (more binding slows wicking)
    flow_t[:n_pos] = rng.normal(420, 30, n_pos)  # ~7 min
    flow_t[n_pos:] = rng.normal(390, 25, n_neg)  # ~6.5 min

    # Add measurement noise
    T_line += rng.normal(0, noise_std, N)
    C_line += rng.normal(0, noise_std, N)
    BG     += rng.normal(0, noise_std/2, N)

    # Clip to physical range [0, 1]
    T_line = np.clip(T_line, 0, 1)
    C_line = np.clip(C_line, 1e-3, 1)
    BG     = np.clip(BG, 0, 1)

    ratio  = T_line / C_line
    T_C_diff = T_line - C_line
    flow_norm = (flow_t - 400) / 30   # normalize

    # 6 features
    X = np.stack([T_line, C_line, BG, ratio, T_C_diff, flow_norm], axis=1)

    # shuffle
    idx = rng.permutation(N)
    return X[idx].astype(np.float32), labels[idx].astype(np.float32)

X, y = make_lfa_data(N, fpos, noise, rng)

print(f"  Generated {N} samples ({int(N*fpos)} positive, {N-int(N*fpos)} negative)")
print(f"  Features: T_line, C_line, BG, ratio, T_C_diff, flow_norm")
print(f"  Feature stats:")
feat_names = ["T_line ", "C_line ", "BG     ", "ratio  ", "T_C_diff", "flow_norm"]
for i, name in enumerate(feat_names):
    pos_m = X[y==1, i].mean()
    neg_m = X[y==0, i].mean()
    print(f"    {name}: pos_mean={pos_m:.4f}  neg_mean={neg_m:.4f}  sep={abs(pos_m-neg_m):.4f}")
print()

# ============================================================
# 2. PYTORCH DATASET + AUGMENTATION
# ============================================================
class LFADataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.02):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self):  return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # Gaussian noise augmentation (simulates sensor variation)
            x += torch.randn_like(x) * self.noise_std
            x = x.clamp(0, 2)  # physical bounds
        return x, self.y[i]

dataset = LFADataset(X, y, augment=True, noise_std=0.02)
n_val   = int(len(dataset) * args.val_frac)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(args.seed))

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False)

print(f"=== 2. DATASET SPLIT ===")
print(f"  Train: {n_train}  Val: {n_val}  Augmentation: Gaussian noise std=0.02")
print()

# ============================================================
# 3. MODEL DEFINITION
# ============================================================
class LFAClassifier(nn.Module):
    def __init__(self, n_features=6, hidden=32, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden//2, 1),
            # No sigmoid here: use BCEWithLogitsLoss (numerically stable)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).numpy()

model = LFAClassifier(hidden=args.hidden, dropout=args.dropout)
n_params = sum(p.numel() for p in model.parameters())
print(f"=== 3. MODEL ===")
print(f"  LFAClassifier: input(6) -> Dense({args.hidden}) -> BN -> ReLU -> "
      f"Dropout({args.dropout}) -> Dense({args.hidden//2}) -> BN -> ReLU -> Dense(1)")
print(f"  Parameters: {n_params}  (fits on any MCU with >4KB RAM)")
print()

# ============================================================
# 4. TRAINING
# ============================================================
print(f"=== 4. TRAINING ({args.epochs} epochs, lr={args.lr}) ===")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def evaluate(model, dl):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            logits = model(xb)
            loss   = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            preds  = (torch.sigmoid(logits) > args.threshold).float()
            correct += (preds == yb).sum().item()
            total  += len(yb)
    return total_loss/total, correct/total

train_losses, val_accs = [], []
for epoch in range(args.epochs):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
        tr_loss, tr_acc = evaluate(model, train_dl)
        vl_loss, vl_acc = evaluate(model, val_dl)
        train_losses.append(tr_loss)
        val_accs.append(vl_acc)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_acc={vl_acc:.4f}")

print()

# ============================================================
# 5. DECISION: SEND ONE PERSON HOME?
# ============================================================
print("=== 5. DECISION: positive -> send person home ===")
print(f"  Threshold: {args.threshold}  (tune for sensitivity vs specificity)")
print()

# Final validation metrics
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_dl:
        probs = torch.sigmoid(model(xb)).numpy()
        all_probs.extend(probs)
        all_labels.extend(yb.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
preds      = (all_probs >= args.threshold).astype(float)

TP = ((preds==1) & (all_labels==1)).sum()
TN = ((preds==0) & (all_labels==0)).sum()
FP = ((preds==1) & (all_labels==0)).sum()
FN = ((preds==0) & (all_labels==1)).sum()

sens     = TP / (TP+FN) if (TP+FN)>0 else 0   # recall / sensitivity
spec     = TN / (TN+FP) if (TN+FP)>0 else 0   # specificity
ppv      = TP / (TP+FP) if (TP+FP)>0 else 0   # precision / PPV
npv      = TN / (TN+FN) if (TN+FN)>0 else 0   # NPV
accuracy = (TP+TN) / len(preds)

print(f"  Confusion matrix (val set, n={len(preds)}):")
print(f"                 Predicted-  Predicted+")
print(f"  Actual neg:    TN={TN:5.0f}       FP={FP:5.0f}")
print(f"  Actual pos:    FN={FN:5.0f}       TP={TP:5.0f}")
print()
print(f"  Sensitivity (recall):  {sens:.4f}  (miss rate = {1-sens:.4f})")
print(f"  Specificity:           {spec:.4f}  (false alarm = {1-spec:.4f})")
print(f"  PPV (precision):       {ppv:.4f}")
print(f"  NPV:                   {npv:.4f}")
print(f"  Accuracy:              {accuracy:.4f}")
print()

# Simulate 5 new patients
print("  Individual decisions (5 new samples):")
print(f"  {'Sample':8s}  {'T_line':8s}  {'ratio':8s}  {'P(pos)':8s}  {'Decision':14s}  {'Action'}")
test_cases = [
    ("Alice",  [0.82, 0.63, 0.03, 0.82/0.63, 0.82-0.63, 0.7]),   # strong positive
    ("Bob",    [0.10, 0.67, 0.02, 0.10/0.67, 0.10-0.67,-0.3]),   # clear negative
    ("Carol",  [0.51, 0.60, 0.04, 0.51/0.60, 0.51-0.60, 0.1]),   # borderline
    ("Dave",   [0.08, 0.64, 0.03, 0.08/0.64, 0.08-0.64,-0.4]),   # negative
    ("Eve",    [0.79, 0.61, 0.05, 0.79/0.61, 0.79-0.61, 0.5]),   # positive
]
model.eval()
for name, feats in test_cases:
    xt    = torch.tensor([feats], dtype=torch.float32)
    prob  = torch.sigmoid(model(xt)).item()
    pos   = prob >= args.threshold
    action = "SEND HOME (isolate)" if pos else "cleared, return"
    print(f"  {name:8s}  {feats[0]:8.3f}  {feats[3]:8.3f}  {prob:8.4f}  "
          f"{'POSITIVE' if pos else 'negative':14s}  {action}")
print()

# Threshold analysis
print("  Threshold sweep (sensitivity/specificity tradeoff):")
print(f"  {'Threshold':10s}  {'Sens':8s}  {'Spec':8s}  {'FP':6s}  {'FN':6s}  {'note'}")
for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    p = (all_probs >= thr).astype(float)
    tp = ((p==1) & (all_labels==1)).sum()
    tn = ((p==0) & (all_labels==0)).sum()
    fp = ((p==1) & (all_labels==0)).sum()
    fn = ((p==0) & (all_labels==1)).sum()
    sn = tp/(tp+fn) if (tp+fn)>0 else 0
    sp = tn/(tn+fp) if (tn+fp)>0 else 0
    note = ("miss cases"  if thr > 0.65 else
            "false alarms" if thr < 0.35 else
            "balanced" if 0.45<=thr<=0.55 else "")
    print(f"  {thr:10.2f}  {sn:8.4f}  {sp:8.4f}  {fp:6.0f}  {fn:6.0f}  {note}")
print()

# ============================================================
# 6. ONNX EXPORT: MCU / RASPBERRY PI DEPLOYMENT
# ============================================================
print("=== 6. EDGE DEPLOYMENT: ONNX for RPi/MCU ===")
print("""
  GS CONNECTION:
    Fiber evanescent sensor measures I1(t), I2(t)
    GS recovers phase shift delta_phi from binding event
    LFA classifier reads [T_line, C_line, ratio, ...] features
    Extracted from fiber sensor signal in real time

  ONNX EXPORT (to run on RPi CM4 or STM32H7):
    import onnx, onnxruntime
    dummy = torch.zeros(1, 6)
    torch.onnx.export(model, dummy, "lfa_model.onnx",
                      input_names=["features"],
                      output_names=["logit"],
                      opset_version=13)

  INFERENCE ON EDGE (Python):
    sess = onnxruntime.InferenceSession("lfa_model.onnx")
    logit = sess.run(None, {"features": x_np})[0]
    prob  = 1 / (1 + exp(-logit))
    decision = "POSITIVE" if prob > threshold else "negative"

  INFERENCE ON MCU (C, fixed-point):
    // Only 3 layers, 32 weights each -> fits in 4KB
    // Replace sigmoid with lookup table (from _repl_lut_sincos.py)
    float lfa_infer(float features[6]) {
        float h1[32], h2[16];
        matmul(W1, features, h1, 32, 6);  // layer 1
        relu(h1, 32);
        matmul(W2, h1, h2, 16, 32);       // layer 2
        relu(h2, 16);
        float logit = dot(W3, h2, 16);    // output
        return 1.0f / (1.0f + expf(-logit));
    }

  LATENCY ESTIMATE:
    RPi CM4 (1.5 GHz ARM):  ~0.05 ms inference  (20,000 tests/s)
    STM32H7 (480 MHz):      ~1.0 ms inference   (1,000 tests/s)
    vs lab flow cytometer: 1000 cells/s (matched)
""")

# Quick model size estimate
total_bytes = sum(p.numel() * 4 for p in model.parameters())
print(f"  Model size: {n_params} params x 4 bytes = {total_bytes} bytes = {total_bytes/1024:.1f} KB")
print(f"  Fits on: RPi CM4 (4GB), STM32H7 (1MB flash), Arduino Portenta (8MB)")
print(f"  Too large for: ATmega328 (32KB) -> need quantization to int8")
print()
print(f"  Int8 quantization: {total_bytes//4} bytes = {total_bytes//4/1024:.2f} KB -> fits on most MCUs")
print()

# Summary table
print("=" * 62)
print("SUMMARY")
print("=" * 62)
print(f"  Model trained on {N} synthetic LFA samples ({int(N*fpos)} pos)")
print(f"  Val accuracy:    {accuracy:.1%}")
print(f"  Sensitivity:     {sens:.1%}  (fraction of positives caught)")
print(f"  Specificity:     {spec:.1%}  (fraction of negatives cleared)")
print(f"  Model size:      {total_bytes} bytes ({total_bytes/1024:.1f} KB)")
print(f"  Decision rule:   P(pos) >= {args.threshold} -> POSITIVE -> send home")
print()
print("  Physics chain:")
print("    fiber evanescent field -> delta_n from binding")
print("    -> GS phase recovery -> intensity features")
print("    -> LFAClassifier -> P(pos) -> binary decision")
print("    all on ONE fiber strand + RPi CM4 = RogueGuard + LOC")
