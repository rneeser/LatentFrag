"""
Test script to verify that PyKeOps → PyTorch replacements produce identical outputs.

Usage:
  1. Run with current PyKeOps code to capture reference outputs:
       python tests/test_pykeops_replacement.py --save-reference
  2. After replacing PyKeOps code, run to verify equivalence:
       python tests/test_pykeops_replacement.py --check
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REF_DIR = os.path.join(os.path.dirname(__file__), "reference_outputs")


def make_test_data(seed=42):
    """Create deterministic test data simulating a small protein surface."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Simulate 2 batch elements: 150 and 100 surface points
    N1, N2 = 150, 100
    N = N1 + N2

    # Surface points (roughly on a sphere of radius ~10 Å)
    vertices = torch.randn(N, 3, device=DEVICE) * 5.0
    batch = torch.cat([
        torch.zeros(N1, dtype=torch.long, device=DEVICE),
        torch.ones(N2, dtype=torch.long, device=DEVICE),
    ])

    # Normals (unit vectors)
    normals = F.normalize(torch.randn(N, 3, device=DEVICE), dim=-1)

    # Atom centers (more atoms than surface points)
    M1, M2 = 200, 160
    M = M1 + M2
    atoms = torch.randn(M, 3, device=DEVICE) * 4.0
    batch_atoms = torch.cat([
        torch.zeros(M1, dtype=torch.long, device=DEVICE),
        torch.ones(M2, dtype=torch.long, device=DEVICE),
    ])

    # Atom types: 6-dim one-hot
    atomtypes = torch.zeros(M, 6, device=DEVICE)
    for i in range(M):
        atomtypes[i, torch.randint(0, 6, (1,)).item()] = 1.0

    return {
        "vertices": vertices,
        "batch": batch,
        "normals": normals,
        "atoms": atoms,
        "batch_atoms": batch_atoms,
        "atomtypes": atomtypes,
    }


def test_grid_cluster(data):
    """Test the subsample function."""
    from latentfrag.encoder.dmasif.geometry_processing import subsample

    # Single batch
    pts = data["vertices"][:150]
    result_single = subsample(pts, batch=None, scale=1.0)

    # Multi batch
    result_multi_pts, result_multi_batch = subsample(
        data["vertices"], batch=data["batch"], scale=1.0
    )

    return {
        "subsample_single": result_single,
        "subsample_multi_pts": result_multi_pts,
        "subsample_multi_batch": result_multi_batch,
    }


def test_soft_distances(data):
    """Test the soft_distances function."""
    from latentfrag.encoder.dmasif.geometry_processing import soft_distances

    # Without atom types
    result_no_types = soft_distances(
        data["atoms"], data["vertices"],
        data["batch_atoms"], data["batch"],
        smoothness=0.5, atomtypes=None,
    )

    # With atom types
    result_with_types = soft_distances(
        data["atoms"], data["vertices"],
        data["batch_atoms"], data["batch"],
        smoothness=0.01, atomtypes=data["atomtypes"],
    )

    return {
        "soft_dist_no_types": result_no_types,
        "soft_dist_with_types": result_with_types,
    }


def test_mesh_normals_areas(data):
    """Test the mesh_normals_areas function."""
    from latentfrag.encoder.dmasif.geometry_processing import mesh_normals_areas

    # Single scale
    normals_single, _ = mesh_normals_areas(
        data["vertices"], normals=data["normals"],
        scale=[0.5], batch=data["batch"],
    )

    # Multi scale
    normals_multi, _ = mesh_normals_areas(
        data["vertices"], normals=data["normals"],
        scale=[0.5, 1.0, 2.0], batch=data["batch"],
    )

    return {
        "normals_single": normals_single,
        "normals_multi": normals_multi,
    }


def test_curvatures(data):
    """Test the curvatures function."""
    from latentfrag.encoder.dmasif.geometry_processing import curvatures

    result = curvatures(
        data["vertices"], normals=data["normals"],
        scales=[1.0, 2.0], batch=data["batch"],
    )

    return {"curvatures": result}


def test_knn_atoms(data):
    """Test the knn_atoms function."""
    from latentfrag.encoder.dmasif.model import knn_atoms

    idx, dists = knn_atoms(
        data["vertices"], data["atoms"],
        data["batch"], data["batch_atoms"],
        k=16,
    )

    return {"knn_idx": idx, "knn_dists": dists}


def test_dmasif_conv(data):
    """Test the full dMaSIFConv_seg + dMaSIFConv forward pass."""
    from latentfrag.encoder.dmasif.geometry_processing import (
        tangent_vectors, dMaSIFConv,
    )
    from latentfrag.encoder.dmasif.model import dMaSIFConv_seg
    from latentfrag.encoder.dmasif.helper import ranges_slices

    torch.manual_seed(123)
    N = data["vertices"].shape[0]

    # Create a small dMaSIFConv_seg
    conv_seg = dMaSIFConv_seg(
        in_channels=8, hidden_channels=16, out_channels=16, n_layers=2, radius=9.0
    ).to(DEVICE)

    # Simulate weights (orientation scores)
    weights = torch.randn(N, 1, device=DEVICE)

    # Load mesh
    conv_seg.load_mesh(
        data["vertices"],
        normals=data["normals"],
        weights=weights,
        batch=data["batch"],
    )

    # Forward with random features
    features = torch.randn(N, 8, device=DEVICE)
    output = conv_seg(features)

    return {
        "conv_nuv": conv_seg.nuv.detach(),
        "conv_output": output.detach(),
    }


def test_dmasif_conv_layer(data):
    """Test a single dMaSIFConv layer (the core convolution)."""
    from latentfrag.encoder.dmasif.geometry_processing import (
        tangent_vectors, dMaSIFConv,
    )
    from latentfrag.encoder.dmasif.helper import ranges_slices

    torch.manual_seed(456)
    N = data["vertices"].shape[0]

    # Create dMaSIFConv layers (cheap and non-cheap)
    conv_cheap = dMaSIFConv(in_channels=8, out_channels=16, radius=9.0, cheap=True).to(DEVICE)
    conv_full = dMaSIFConv(in_channels=8, out_channels=16, radius=9.0, cheap=False).to(DEVICE)

    # nuv: (N, 3, 3)
    normals = data["normals"]
    tangent_bases = tangent_vectors(normals)  # (N, 2, 3)
    nuv = torch.cat([normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)], dim=1)  # (N, 3, 3)

    ranges_info, _ = ranges_slices(data["batch"])
    ranges = (ranges_info,)  # Wrap in tuple for dMaSIFConv.forward API
    features = torch.randn(N, 8, device=DEVICE)

    out_cheap = conv_cheap(data["vertices"], nuv, features, ranges)
    out_full = conv_full(data["vertices"], nuv, features, ranges)

    return {
        "conv_layer_cheap": out_cheap.detach(),
        "conv_layer_full": out_full.detach(),
    }


ALL_TESTS = {
    "grid_cluster": test_grid_cluster,
    "soft_distances": test_soft_distances,
    "mesh_normals_areas": test_mesh_normals_areas,
    "curvatures": test_curvatures,
    "knn_atoms": test_knn_atoms,
    "dmasif_conv": test_dmasif_conv,
    "dmasif_conv_layer": test_dmasif_conv_layer,
}


def save_reference():
    """Run all tests and save reference outputs."""
    os.makedirs(REF_DIR, exist_ok=True)
    data = make_test_data()

    for name, test_fn in ALL_TESTS.items():
        print(f"Running {name}...", end=" ", flush=True)
        results = test_fn(data)
        ref_path = os.path.join(REF_DIR, f"{name}.pt")
        # Move all tensors to CPU for storage
        results_cpu = {k: v.cpu() for k, v in results.items()}
        torch.save(results_cpu, ref_path)
        print(f"saved {len(results)} tensors")

    print(f"\nReference outputs saved to {REF_DIR}/")


def check_equivalence():
    """Run all tests and compare against reference outputs."""
    if not os.path.exists(REF_DIR):
        print(f"ERROR: Reference directory {REF_DIR} not found. Run with --save-reference first.")
        sys.exit(1)

    data = make_test_data()
    all_passed = True

    for name, test_fn in ALL_TESTS.items():
        ref_path = os.path.join(REF_DIR, f"{name}.pt")
        if not os.path.exists(ref_path):
            print(f"  SKIP {name}: no reference file")
            continue

        print(f"Testing {name}...", end=" ", flush=True)
        load_kwargs = {}
        if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (1, 13):
            load_kwargs['weights_only'] = True
        ref = torch.load(ref_path, **load_kwargs)

        try:
            results = test_fn(data)
        except Exception as e:
            print(f"FAIL (exception: {e})")
            all_passed = False
            continue

        test_passed = True
        for key in ref:
            ref_val = ref[key].to(DEVICE)
            new_val = results[key]

            if ref_val.dtype in (torch.long, torch.int, torch.int32):
                match = torch.equal(ref_val, new_val)
                if not match:
                    diff_count = (ref_val != new_val).sum().item()
                    print(f"\n    {key}: {diff_count}/{ref_val.numel()} elements differ", end="")
                    test_passed = False
            else:
                # Float comparison with tolerance
                if not torch.allclose(ref_val, new_val, atol=1e-5, rtol=1e-4):
                    max_diff = (ref_val - new_val).abs().max().item()
                    mean_diff = (ref_val - new_val).abs().mean().item()
                    print(f"\n    {key}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", end="")
                    test_passed = False

        if test_passed:
            print("PASS")
        else:
            print("\n  -> FAIL")
            all_passed = False

    if all_passed:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save-reference", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.save_reference:
        save_reference()
    else:
        check_equivalence()
