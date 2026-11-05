"""Check EDLHead outputs on ten EEGNet trials."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn


def try_import_edl():
    try:
        from models.edl_head import EDLHead
        return EDLHead
    except Exception:
        try:
            from CalibMI.models.edl_head import EDLHead  # type: ignore
            return EDLHead
        except Exception as e:
            print("ERROR: could not import EDLHead (models/edl_head.py).", e)
            raise


def build_eegnet_backbone(X_sample, n_classes=4):
    """Construct the active EEGNet backbone.

    Returns an nn.Module backbone that accepts input shaped [B, 1, C, T].
    The returned backbone should provide a `get_latent(x)` method returning [B, 128].
    """
    try:
        from eegnet import _build_eegnet
        torch_mod = torch
        nn_mod = nn
        n_channels = int(X_sample.shape[1])
        n_times = int(X_sample.shape[2])
        model = _build_eegnet(torch_mod, nn_mod, n_channels, n_times, n_classes, dropout=0.5)

        if hasattr(model, 'get_latent'):
            return model

        if hasattr(model, 'features'):
            class BackboneWrapper(nn.Module):
                def __init__(self, model, latent_dim=128):
                    super().__init__()
                    self.model = model
                    with torch.no_grad():
                        dummy = torch.zeros(1, 1, n_channels, n_times)
                        feat = self.model.features(dummy)
                        feat_dim = int(feat.reshape(1, -1).shape[1])
                    self._proj = nn.Linear(feat_dim, latent_dim)

                def forward(self, x):
                    return self.model(x)

                def get_latent(self, x):
                    f = self.model.features(x)
                    f = f.view(f.size(0), -1)
                    return self._proj(f)

            return BackboneWrapper(model)

        return model
    except Exception as e:
        print("Could not construct EEGNet backbone via src/eegnet.py:", e)
        raise


def load_checkpoint_to_model(model, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        state = ck['state_dict']
    else:
        state = ck
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint into backbone (strict=False).")
    except Exception as e:
        print("Warning: loading state_dict with strict=False failed; attempting naive assignment.", e)
    return model


def run_validation(args):
    EDLHead = try_import_edl()

    npz_path = args.data_path
    if not os.path.exists(npz_path):
        print(f"Data file not found: {npz_path}")
        sys.exit(1)
    data = np.load(npz_path)
    if 'X' not in data or 'y' not in data:
        print(f"NPZ file missing required keys 'X' and 'y'. Found: {list(data.keys())}")
        sys.exit(1)
    X = data['X']

    rng = np.random.default_rng(seed=0)
    N = X.shape[0]
    if N < 10:
        raise ValueError(f"Not enough trials in {npz_path}: found {N}")
    idx = rng.choice(N, size=10, replace=False)
    X_sel = X[idx]
    if X_sel.ndim == 3:
        X_t = torch.tensor(X_sel[:, None, :, :], dtype=torch.float32)
    elif X_sel.ndim == 4:
        X_t = torch.tensor(X_sel, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected X shape: {X_sel.shape}")

    try:
        backbone = build_eegnet_backbone(X_sel)
    except Exception:
        print("Failed to build EEGNet backbone. Aborting.")
        sys.exit(1)

    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        try:
            backbone = load_checkpoint_to_model(backbone, ckpt_path)
        except Exception as e:
            print("Warning: failed to load checkpoint:", e)
    else:
        print(f"Checkpoint not found at {ckpt_path}; continuing with randomly initialized backbone.")

    edl = EDLHead(in_dim=128, num_classes=args.num_classes)

    class ModelWithEDL(nn.Module):
        def __init__(self, backbone, edl_head):
            super().__init__()
            self.backbone = backbone
            self.edl = edl_head

        def forward(self, x):
            if hasattr(self.backbone, 'get_latent'):
                z = self.backbone.get_latent(x)
            else:
                if hasattr(self.backbone, 'features'):
                    f = self.backbone.features(x)
                    f = f.view(f.size(0), -1)
                    if f.size(1) != 128:
                        proj = nn.Linear(f.size(1), 128).to(f.device)
                        z = proj(f)
                    else:
                        z = f
                else:
                    out = self.backbone(x)
                    if out.dim() == 2 and out.size(1) != 128:
                        proj = nn.Linear(out.size(1), 128).to(out.device)
                        z = proj(out)
                    else:
                        z = out
            return self.edl(z)

    model = ModelWithEDL(backbone, edl)
    model.eval()

    with torch.no_grad():
        out = model(X_t)

    alpha = out['alpha'].cpu().numpy()
    p_hat = out['p_hat'].cpu().numpy()
    vacuity = out['vacuity'].cpu().numpy()
    fail = False

    vac_mean = float(vacuity.mean())
    vac_std = float(vacuity.std())
    print(f"Vacuity mean: {vac_mean:.6f}, std: {vac_std:.6f}")

    in_range = np.all((vacuity >= -1e-8) & (vacuity <= 1.0 + 1e-8))
    if in_range:
        print("CHECK vacuity in [0,1]: PASS")
    else:
        fail = True
        bad_idx = np.where((vacuity < -1e-8) | (vacuity > 1.0 + 1e-8))[0]
        print("CHECK vacuity in [0,1]: FAIL. Bad values:")
        for i in bad_idx:
            print(f" idx={i}, vacuity={vacuity[i]:.6f}")

    alpha_gt1 = np.all(alpha > 1.0 - 1e-12)
    if alpha_gt1:
        print("CHECK alpha > 1.0: PASS")
    else:
        fail = True
        bad = np.where(~(alpha > 1.0 - 1e-12))
        print("CHECK alpha > 1.0: FAIL. Examples:")
        for idx in zip(bad[0][:5], bad[1][:5]):
            print(f" trial={idx[0]}, class={idx[1]}, alpha={alpha[idx]:.6f}")

    sums = p_hat.sum(axis=1)
    ok = np.allclose(sums, 1.0, atol=1e-5)
    if ok:
        print("CHECK p_hat sums to 1.0: PASS")
    else:
        fail = True
        bad_idx = np.where(~np.isclose(sums, 1.0, atol=1e-5))[0]
        print("CHECK p_hat sums to 1.0: FAIL. Examples:")
        for i in bad_idx:
            print(f" idx={i}, sum={sums[i]:.8f}, p_hat={p_hat[i]}")

    if fail:
        print("VALIDATION: FAIL (one or more checks failed)")
        sys.exit(2)
    else:
        print("VALIDATION: PASS (all checks passed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v2_best.pt')
    parser.add_argument('--data-path', type=str, default='data/bci4_2a_subject1_train.npz')
    parser.add_argument('--num-classes', type=int, default=4)
    args = parser.parse_args()
    run_validation(args)


if __name__ == '__main__':
    main()
