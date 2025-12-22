#mine12 fully reversible, testing xyz, reversible for the latent image only


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

#======================================
# A distance aware bce loss function
#======================================
def distance_aware_bce_loss(logits, targets):
    """
    logits: [batch, 1024]
    targets: [batch, 1024] with multiple possible 1s
    """
    B, L = targets.shape
    device = targets.device

    # Index positions 0..L-1
    idx = torch.arange(L, device=device).float().unsqueeze(0).expand(B, L)

    # Mask of positive bits
    true_positions = (targets > 0).float()

    # Identify samples that have at least one positive bit
    has_pos = true_positions.sum(dim=1) > 0

    distances = torch.zeros_like(idx)

    for b in range(B):
        if has_pos[b]:
            pos = torch.where(true_positions[b] == 1)[0].float()  # [num_pos]
            dist = torch.abs(idx[b].unsqueeze(1) - pos.unsqueeze(0))  # [L, num_pos]
            min_dist, _ = torch.min(dist, dim=1)
            distances[b] = min_dist
        else:
            # No positives → neutral weight
            distances[b] = 1.0

    # Convert distance to weights
    weights = 1.0 / (1.0 + distances)

    # Elementwise BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")

    # Weighted loss
    return (bce * weights).mean()

# ============================================================
#  SPACE-TO-DEPTH (invertible)
# ============================================================

class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.reshape(B, C * self.bs * self.bs, H // self.bs, W // self.bs)

    def inverse(self, x):
        B, C, H, W = x.shape
        bs2 = self.bs * self.bs
        x = x.view(B, C // bs2, self.bs, self.bs, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(B, C // bs2, H * self.bs, W * self.bs)


# ============================================================
#  REVERSIBLE COUPLING BLOCK
# ============================================================

class RevBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c1 = channels // 2
        c2 = channels - c1

        def net(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1)
            )

        self.F = net(c1, c2)
        self.G = net(c2, c1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y2 = x2 + self.F(x1)
        y1 = x1 + self.G(y2)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        x1 = y1 - self.G(y2)
        x2 = y2 - self.F(x1)
        return torch.cat([x1, x2], dim=1)


# ============================================================
#  INVERTIBLE NETWORK (i-RevNet-like)
# ============================================================

class InvertibleNet(nn.Module):
    def __init__(self, blocks_per_stage=2):
        super().__init__()

        self.transforms = nn.ModuleList()

        # channel progression via squeezing
        channels = [4, 16, 64, 256, 1024]

        for i in range(len(channels) - 1):
            self.transforms.append(SpaceToDepth(2))
            for _ in range(blocks_per_stage):
                self.transforms.append(RevBlock(channels[i + 1]))

    def forward(self, x):
        z = x
        for t in self.transforms:
            z = t(z)
        # z is latent representation (B, 1024)
        return z.view(z.size(0), 1024)

    def reverse(self, z):
        x = z.view(z.size(0), 1024, 1, 1)
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x


# ============================================================
#  INVERTIBILITY VALIDATION
# ============================================================

def validate_invertibility(model, x):
    with torch.no_grad():
        z = model(x)
        xr = model.reverse(z)

    diff = (x - xr).abs()
    print("\nInvertibility validation:")
    print(f"  Max abs error:  {diff.max().item():.6e}")
    print(f"  Mean abs error: {diff.mean().item():.6e}")


# ============================================================
#  TRAINING LOOP
# ============================================================

def train(model, X, Y, Xv, Yv, epochs=5, batch_size=32, lr=1e-3):
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = distance_aware_bce_loss

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        losses = []

        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]

            opt.zero_grad()
            Zb = model(xb)          # latent
            assert Zb.shape == yb.shape, "Z and Y must have same shape"
            loss = loss_fn(Zb, yb)

            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            Zv = model(Xv)
            preds = (Zv > 0).float()
            val_loss = loss_fn(Zv, Yv).item()
            preds = (Zv > 0).float()
            f1 = f1_score(Yv.cpu(), preds.cpu(), average="macro",  zero_division=0)

        print(f"Epoch {ep}/{epochs} | "
              f"Train={np.mean(losses):.4f} | "
              f"Val={val_loss:.4f} | "
              f"F1={f1:.4f}")

        #some better output at intervals
        if ep % 20 == 0:  # change to print more/less often
            pred = (torch.sigmoid(Zv[0]) > 0.5).int()
            true = Yv[0].int()

            pred_idx = torch.where(pred == 1)[0].tolist()
            true_idx = torch.where(true == 1)[0].tolist()

            print("\nSample validation result:")
            print(f"Predicted 1s: {pred_idx}")
            print(f"True 1s:      {true_idx}")

            print("=============================================\n")

#=======================================
#Main function
#=======================================
if __name__ == "__main__":
    #read the csv data
    df = pd.read_csv('reversibledata.csv')
    X = df["C"]
    y = df["D"]
    yarr=list(y)
    Xnew=[]
    Ynew=[]
    for xloc,id in zip(X,df["A"]):
        newx=[]
        newx.append([])
        newx.append([])
        newx.append([])
        newx.append([])
        xarr=list(xloc)
        count=0
        for i in range(16):
            row=[]
            row.append([])
            row.append([])
            row.append([])
            row.append([])
            for k in range(16):
                num=0
                if k>=i:
                    num=ord(xarr[count]) - ord('0')
                if num>5:
                    row[3].append(1)
                    num=num-5
                else:
                    row[3].append(0)
                for l in range(3):
                    if num!=0 and num==l+1:
                        row[l].append(1)
                    else:
                        row[l].append(0)
                if k>=i:
                     count=count+1
            newx[0].append(row[0])
            newx[1].append(row[1])
            newx[2].append(row[2])
            newx[3].append(row[3])
        Xnew.append(newx)

    for yloc in y:
        newy=list(yloc)
        newnewy=[]
        for c in newy:
            newnewy.append(ord(c) - ord('0'))
        Ynew.append(newnewy)

    #check some dimensions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(len(Xnew),len(Xnew[0]), len(Xnew[0][0]))
    targets = torch.tensor(Ynew, dtype=torch.float32).to(device)
    inputs = torch.tensor(Xnew, dtype=torch.float32).to(device)
    print(inputs.size(),targets.size())

    #split 80:20
    val_split=0.2
    targets = targets.float()
    N = inputs.size(0)
    idx = torch.randperm(N)
    split = int(N * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_inputs, train_targets = inputs[train_idx], targets[train_idx]
    val_inputs, val_targets = inputs[val_idx], targets[val_idx]

    #train the model
    model = InvertibleNet().cuda()
    train(model,
                        train_inputs, train_targets,
                        val_inputs, val_targets,
                        epochs=121)
    
    # Invertibility validation on REAL validation sample
    print("\nValidating invertibility on an actual validation example...")
    sample = val_inputs[0:1].cuda()   # first validation sample
    validate_invertibility(model, sample)
    print("\n=== Diagnostic: trained Z vs random Z ===")
    model.eval()
    with torch.no_grad():
        # Trained latent -> reverse -> forward
        x_real = val_inputs[0:1]                 # real input
        Z_real = model(x_real)              # latent from data
        x_rec = model.reverse(Z_real)       # reconstruction
        Z_rec = model(x_rec)                # latent again

        x_err = (x_real - x_rec).abs()
        z_err = (Z_real - Z_rec).abs()

        print("\n[Trained Z]")
        print(f"X reconstruction max error : {x_err.max().item():.6e}")
        print(f"X reconstruction mean error: {x_err.mean().item():.6e}")
        print(f"Z cycle max error          : {z_err.max().item():.6e}")
        print(f"Z cycle mean error         : {z_err.mean().item():.6e}")

        # Random latent -> reverse
        Z_rand = torch.randn_like(Z_real)
        x_rand = model.reverse(Z_rand)

        print("\n[Random Z]")
        print("Random Z stats:")
        print(f"  mean={Z_rand.mean().item():.4f}, std={Z_rand.std().item():.4f}")
        print("Generated X stats:")
        print(f"  mean={x_rand.mean().item():.4f}, std={x_rand.std().item():.4f}")
        print(f"  min={x_rand.min().item():.4f}, max={x_rand.max().item():.4f}")





