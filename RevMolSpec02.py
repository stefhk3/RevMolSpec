import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

# ============================================================
#  Various tests to run on the network
# ============================================================
@torch.no_grad()
def test_conditional_variation(model, Y_fixed, n=8):
    model.eval()

    Zf = torch.randn(n, model.z_free_dim, device=Y_fixed.device)
    Y = Y_fixed.expand(n, -1)

    X = model.inverse(Zf, Y)

    diffs = (X[1:] - X[:-1]).abs().mean(dim=(1,2,3))
    print("Mean X diffs under Z_free variation:", diffs.tolist())


@torch.no_grad()
def test_random_Y_generation(model, n=4):
    model.eval()

    Zf = torch.randn(n, model.z_free_dim, device=next(model.parameters()).device)
    Yl = torch.randn(n, model.y_dim, device=Zf.device)

    X_gen = model.inverse(Zf, Yl)
    Zf2, Yl2 = model(X_gen)

    print("Generated X range:", X_gen.min().item(), X_gen.max().item())
    print("Z_free diff:", (Zf - Zf2).abs().mean().item())
    print("Y_latent diff:", (Yl - Yl2).abs().mean().item())

@torch.no_grad()
def test_latent_consistency(model, X):
    model.eval()

    Zf1, Yl1 = model(X)
    X_rec = model.inverse(Zf1, Yl1)
    Zf2, Yl2 = model(X_rec)

    z_err = (Zf1 - Zf2).abs().mean().item()
    y_err = (Yl1 - Yl2).abs().mean().item()

    print(f"[Z_free round-trip error] {z_err:.3e}")
    print(f"[Y_latent round-trip error] {y_err:.3e}")

def validate_invertibility(model, x):
    with torch.no_grad():
        model.eval()
        Zf, Yl = model(x)

    X_rec = model.inverse(Zf, Yl)

    max_err = (x - X_rec).abs().max().item()
    mean_err = (x - X_rec).abs().mean().item()

    print(f"[X→Z→X] max error : {max_err:.3e}")
    print(f"[X→Z→X] mean error: {mean_err:.3e}")

    return max_err

# ============================================================
#  This function validates certain properties after an epoch 
# ============================================================
@torch.no_grad()
def validate_epoch(
    model,
    X_val,
    Y_val,
    coarsen_Y,
    loss_fn,
    max_print=2
):
    model.eval()

    # Forward
    Z_free, Y_latent = model(X_val)
    Zf, Yl = model(X_val[:2])
    print("Z_free shape:", Zf.shape)
    print("Y_latent shape:", Yl.shape)

    # ---- Loss ----
    Y_target_coarse = coarsen_Y(Y_val, y_coarse_dim=128)
    val_loss = loss_fn(Y_latent, Y_target_coarse).item()
    print("Target coarse mean:", Y_target_coarse.mean().item())
    # ---- F1 ----
    y_true = Y_target_coarse.cpu().numpy().astype(int)
    y_pred = (Y_latent > 0).cpu().numpy().astype(int)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ---- Z_free stats ----
    z_mean = Z_free.mean().item()
    z_std = Z_free.std().item()
    z_min = Z_free.min().item()
    z_max = Z_free.max().item()

    # ---- Invertibility test (true latent only!) ----
    X_recon = model.inverse(Z_free, Y_latent)
    inv_error = (X_recon - X_val).abs().max().item()

    # ---- Optional qualitative check ----
    if max_print > 0:
        print("\nSample Y comparison:")
        for i in range(min(max_print, X_val.shape[0])):
            print("origiN sum:", int(Y_val[i].sum().item()))
            print("target sum:", int(y_true[i].sum().item()))
            print("pred sum  :", int(y_pred[i].sum().item()))
            print("overlap   :", int((y_true[i] * y_pred[i]).sum().item()))
            probs = torch.sigmoid(Y_latent)
            topk = torch.topk(probs[i], k=10)
            print("top probs:", topk.values.tolist())
            print("top idxs :", topk.indices.tolist())

    return {
        "val_loss": val_loss,
        "f1": f1,
        "z_mean": z_mean,
        "z_std": z_std,
        "z_min": z_min,
        "z_max": z_max,
        "inv_error": inv_error,
    }

# ============================================================
#  Functions to squeeze/unsqueze layers
# ============================================================
def squeeze(x):
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    x = x.view(B, C, H//2, 2, W//2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C*4, H//2, W//2)

def unsqueeze(x):
    B, C, H, W = x.shape
    assert C % 4 == 0
    x = x.view(B, C//4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C//4, H*2, W*2)


def estimate_entropy(X):
    p = X.mean(dim=0)
    H = -(p*torch.log2(p+1e-10) + (1-p)*torch.log2(1-p+1e-10)).sum()
    return H.item()

# ============================================================
#  Function to coarsen a vector
# ============================================================
def coarsen_Y(Y, y_coarse_dim=128):
    B, D = Y.shape
    assert D % y_coarse_dim == 0
    g = D // y_coarse_dim
    Yg = Y.view(B, y_coarse_dim, g)
    return Yg.max(dim=2).values

# ============================================================
#  loss function (this is still problematic
# ============================================================
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

class ConditionalIRevNet(nn.Module):
    def __init__(self, input_channels=4, height=16, width=16,
                 y_dim=128, n_blocks=4):
        super().__init__()

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.y_dim = y_dim

        self.total_dim = input_channels * height * width
        assert y_dim <= self.total_dim, "y_dim too large"

        self.z_free_dim = self.total_dim - y_dim

        n1, n2, n3, n4, n5 = 2, 2, 4, 6, 8
        self.stages = nn.ModuleList([
            nn.ModuleList([RevBlock(4)   for _ in range(n1)]),
            nn.ModuleList([RevBlock(16)  for _ in range(n2)]),
            nn.ModuleList([RevBlock(64)  for _ in range(n3)]),
            nn.ModuleList([RevBlock(256) for _ in range(n4)]),
            nn.ModuleList([RevBlock(1024) for _ in range(n5)]),
        ])

    # -----------------
    # Forward: X -> (Z_free, Y_latent)
    # -----------------
    def forward(self, x):
        z = x
        for stage in self.stages[:-1]:
            for block in stage:
                z = block(z)
            z = squeeze(z)

        for block in self.stages[-1]:
            z = block(z)

        # z is now [B, 1024, 1, 1]
        z = z.view(z.size(0), 1024)

        Z_free = z[:, :self.z_free_dim]
        Y_latent = z[:, self.z_free_dim:]

        return Z_free, Y_latent


    def inverse(self, Z_free, Y_latent):
        z = torch.cat([Z_free, Y_latent], dim=1)
        z = z.view(z.size(0), 1024, 1, 1)

        for block in reversed(self.stages[-1]):
            z = block.inverse(z)

        for stage in reversed(self.stages[:-1]):
            z = unsqueeze(z)
            for block in reversed(stage):
                z = block.inverse(z)

        return z




# ============================================================
#  TRAINING LOOP
# ============================================================

def train(model, X, Y, Xv, Yv, epochs=5, batch_size=32, lr=1e-3):
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = distance_aware_bce_loss
    desired_dim=128

    for ep in range(1, epochs + 1):
        print("=============Epoch: ",ep)
        model.train()
        perm = torch.randperm(len(X))
        losses = []

        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], Y[idx]

            opt.zero_grad()
            Z_free, Y_latent = model(xb)
            loss = loss_fn(Y_latent, coarsen_Y(yb, y_coarse_dim=desired_dim)) #.view(B,-1), Y_true.view(B,-1))
            z_reg = 1e-4 * (Z_free ** 2).mean()
            loss = loss + z_reg
            #assert Zb.shape == yb.shape, "Z and Y must have same shape"

            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        metrics = validate_epoch(
            model,
            Xv,
            Yv,
            coarsen_Y,
            loss_fn
        )

        print(
           f"VAL | loss={metrics['val_loss']:.4f} "
           f"F1={metrics['f1']:.4f} "
           f"Z(mean={metrics['z_mean']:.2f}, std={metrics['z_std']:.2f}) "
           f"inv_err={metrics['inv_error']:.2e}"
        )


if __name__ == "__main__":
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
            #print("x ",row)
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
        #print("y ",newnewy)
        Ynew.append(newnewy)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(len(Xnew),len(Xnew[0]), len(Xnew[0][0]))
    targets = torch.tensor(Ynew, dtype=torch.float32).to(device)
    inputs = torch.tensor(Xnew, dtype=torch.float32).to(device)
    print(inputs.size(),targets.size())
    print(estimate_entropy(inputs))

    val_split=0.2
    targets = targets.float()
    N = inputs.size(0)
    idx = torch.randperm(N)
    split = int(N * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_inputs, train_targets = inputs[train_idx], targets[train_idx]
    val_inputs, val_targets = inputs[val_idx], targets[val_idx]

    model = ConditionalIRevNet().cuda()

    train(model,
                        train_inputs, train_targets,
                        val_inputs, val_targets,
                        epochs=5)
    # ---------------------------------------------------------
    # Invertibility validation on REAL validation sample
    # ---------------------------------------------------------
    print("\nValidating invertibility on an actual validation example...")

    sample = val_inputs[0:1].cuda()   # first validation sample

    validate_invertibility(model, sample)
    test_latent_consistency(model, sample)
    test_random_Y_generation(model)
    test_conditional_variation(model,coarsen_Y(val_targets[0:1], y_coarse_dim=128).cuda())
