"""
Build a skeleton tree (MST + RigGS-style pruning/simplification) from
persistent motion nodes.

Independent re-implementation of the general idea behind RigGS's
obtain_skeleton_tree / prune_tree / simplify_tree, adapted to this project's
own data. Only persistent_nodes.pt (from train_persistent_motion_nodes.py)
is used as node/trajectory data -- raw_tracks_3d.pt, any cluster output, any
observation mesh, and UniRig are never read. Every output joint is literally
one of the existing persistent nodes (by index): this script only selects
and connects a subset of what train_persistent_motion_nodes.py produced --
it never invents a position, learns a new id, or re-matches anything
per-frame. Symmetry correction (RigGS relies on semantic left/right labels)
is intentionally skipped, since this pipeline has no such labels; that is
recorded explicitly in report.json rather than silently omitted.

Output
------
outputs/davis/<seq-name>/skeleton/riggs_skeleton/
    skeleton.pt
    candidate_mst.obj
    skeleton.obj
    skeleton_overlay.mp4
    preview_first_frame<NNNNN>.png
    preview_mid_frame<NNNNN>.png
    preview_last_frame<NNNNN>.png
    report.json

Example
-------
    python flow3d/skeleton/extract_riggs_skeleton.py --seq-name camel --overwrite
"""

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # skeleton/ -> flow3d/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import colormaps
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from flow3d.data.casual_dataset import CasualDataset, DavisDataConfig
from flow3d.data.utils import to_serializable

# Not imported: flow3d.scene_model, flow3d.params, flow3d.renderer, flow3d.trainer,
# any checkpoint/Gaussian/motion-basis/cluster code. raw_tracks_3d.pt, any
# observation mesh, and UniRig are never read -- only persistent_nodes.pt.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq-name", type=str, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/davis/default.yaml"))
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-terminal-hops", type=int, default=3)
    parser.add_argument("--simplify-dist-ratio", type=float, default=1.0)
    parser.add_argument("--max-inserts-per-chain", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_config_path(config: Path) -> Path:
    return config if config.is_absolute() else (_REPO_ROOT / config).resolve()


def load_dataset(config_path: Path, seq_name: str) -> tuple[CasualDataset, dict]:
    scene_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = DavisDataConfig(
        root_dir=scene_cfg["data_dir"], seq_name=seq_name,
        load_from_cache=True, **scene_cfg.get("data", {}),
    )
    dataset = CasualDataset(**asdict(data_cfg))
    return dataset, scene_cfg


# ---------------------------------------------------------------------------
# FPS at the template frame (single fixed pose -- no multi-query complication).
# ---------------------------------------------------------------------------

def fps_template(template_node: torch.Tensor, max_candidates: int) -> list[int]:
    M = template_node.shape[0]
    C = min(max_candidates, M)
    centroid = template_node.mean(dim=0)
    seed = int(torch.argmin((template_node - centroid).norm(dim=-1)).item())
    selected = [seed]
    min_dist = (template_node - template_node[seed]).norm(dim=-1)
    min_dist[seed] = float("-inf")
    for _ in range(C - 1):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        d = (template_node - template_node[nxt]).norm(dim=-1)
        min_dist = torch.minimum(min_dist, d)
        min_dist[nxt] = float("-inf")
    return selected


# ---------------------------------------------------------------------------
# Tree utilities over a plain adjacency dict (candidate-local indices).
# ---------------------------------------------------------------------------

def current_degree(adj: dict[int, set]) -> dict[int, int]:
    return {n: len(neighbors) for n, neighbors in adj.items()}


def bfs_parents(root: int, adj: dict[int, set]) -> tuple[dict[int, int | None], list[int]]:
    parent: dict[int, int | None] = {root: None}
    order = [root]
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                order.append(v)
                dq.append(v)
    return parent, order


def tree_distances_from(source: int, adj: dict[int, set], dist: np.ndarray) -> dict[int, float]:
    d = {source: 0.0}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + float(dist[u, v])
                stack.append(v)
    return d


def walk_chain(start_anchor: int, first_node: int, adj: dict[int, set], degree: dict[int, int]):
    """Walk from start_anchor through first_node (degree==2) until hitting a
    non-degree-2 node. Returns (end_anchor, interior_nodes_in_chain_order)."""
    interior = []
    prev, cur = start_anchor, first_node
    while degree[cur] == 2:
        interior.append(cur)
        nxt = next(x for x in adj[cur] if x != prev)
        prev, cur = cur, nxt
    return cur, interior


# ---------------------------------------------------------------------------
# Chain simplification: global worst-deviation-first worklist (order-independent
# result, unlike plain top-down recursive Douglas-Peucker).
# ---------------------------------------------------------------------------

def point_to_segment_dist_over_time(p_t: torch.Tensor, a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    ab = b_t - a_t
    s = ((p_t - a_t) * ab).sum(-1) / ab.pow(2).sum(-1).clamp(min=1e-12)
    closest = a_t + s.clamp(0, 1)[:, None] * ab
    return (p_t - closest).norm(dim=-1)  # (T,)


def simplify_chain(
    nodes: list[int], a: int, b: int, cand_traj: torch.Tensor, threshold: float, max_inserts: int
) -> list[int]:
    if not nodes:
        return []
    segments = [(a, b, nodes)]
    kept: list[int] = []
    while len(kept) < max_inserts:
        best = None  # (deviation, seg_index, node)
        for si, (sa, sb, snodes) in enumerate(segments):
            for n in snodes:
                dev = point_to_segment_dist_over_time(cand_traj[:, n], cand_traj[:, sa], cand_traj[:, sb]).mean()
                dev = float(dev.item())
                if best is None or dev > best[0]:
                    best = (dev, si, n)
        if best is None or best[0] <= threshold:
            break
        _dev, si, node = best
        kept.append(node)
        sa, sb, snodes = segments.pop(si)
        i = snodes.index(node)
        segments.append((sa, node, snodes[:i]))
        segments.append((node, sb, snodes[i + 1:]))
    return kept


def _project_2d(pos_tn: torch.Tensor, Ks: torch.Tensor, w2cs: torch.Tensor) -> torch.Tensor:
    """pos_tn: (T, N, 3); Ks: (T, 3, 3); w2cs: (T, 4, 4) -> (T, N, 2)."""
    pts_cam = torch.einsum("tij,tnj->tni", w2cs, F.pad(pos_tn, (0, 1), value=1.0))[..., :3]
    pts_img = torch.einsum("tij,tnj->tni", Ks, pts_cam)
    return pts_img[..., :2] / torch.clamp(pts_img[..., 2:], min=1e-5)


def write_obj(path: Path, vertices: np.ndarray, edges: list[tuple[int, int]]) -> None:
    with path.open("w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for i, j in edges:
            f.write(f"l {i + 1} {j + 1}\n")


def _joint_color_palette(J: int) -> np.ndarray:
    cmap = colormaps.get_cmap("gist_rainbow")
    colors_rgb = np.asarray([cmap(i / max(J - 1, 1))[:3] for i in range(max(J, 1))])
    return (colors_rgb[:, ::-1] * 255).astype(int)  # BGR


def _render_skeleton_frame(
    dataset: CasualDataset, t: int, proj_np: np.ndarray, edges: list[tuple[int, int]], colors_bgr: np.ndarray
) -> np.ndarray:
    img_rgb = (dataset.get_image(t).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for i, j in edges:
        p1 = tuple(proj_np[t, i].round().astype(int))
        p2 = tuple(proj_np[t, j].round().astype(int))
        cv2.line(img_bgr, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
    for n in range(proj_np.shape[1]):
        p = tuple(proj_np[t, n].round().astype(int))
        cv2.circle(img_bgr, p, 5, tuple(int(c) for c in colors_bgr[n]), -1, cv2.LINE_AA)
    return img_bgr


def save_previews_and_video(
    dataset: CasualDataset, proj: torch.Tensor, edges: list[tuple[int, int]], output_dir: Path, fps: int
) -> tuple[list[Path], Path]:
    T = proj.shape[0]
    J = proj.shape[1]
    colors_bgr = _joint_color_palette(J)
    proj_np = proj.numpy()

    preview_paths = []
    for label, t in [("first", 0), ("mid", T // 2), ("last", T - 1)]:
        img_bgr = _render_skeleton_frame(dataset, t, proj_np, edges, colors_bgr)
        out_path = output_dir / f"preview_{label}_frame{dataset.frame_names[t]}.png"
        cv2.imwrite(str(out_path), img_bgr)
        preview_paths.append(out_path)

    video_path = output_dir / "skeleton_overlay.mp4"
    with imageio.get_writer(video_path, fps=fps) as writer:
        for t in range(T):
            img_bgr = _render_skeleton_frame(dataset, t, proj_np, edges, colors_bgr)
            writer.append_data(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return preview_paths, video_path


def main() -> None:
    args = build_parser().parse_args()

    config_path = resolve_config_path(args.config)
    persistent_nodes_path = (
        _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "persistent_nodes" / "persistent_nodes.pt"
    )
    if not persistent_nodes_path.exists():
        raise FileNotFoundError(
            f"{persistent_nodes_path} not found. Run "
            f"'python flow3d/skeleton/train_persistent_motion_nodes.py --seq-name {args.seq_name}' first."
        )

    output_dir = _REPO_ROOT / "outputs" / "davis" / args.seq_name / "skeleton" / "riggs_skeleton"
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = output_dir / "skeleton.pt"
    if skeleton_path.exists() and not args.overwrite:
        raise FileExistsError(f"{skeleton_path} already exists. Pass --overwrite to replace it.")

    start_time = time.time()
    persistent = torch.load(persistent_nodes_path)
    dataset, _scene_cfg = load_dataset(config_path, args.seq_name)
    trajectory = persistent["trajectory"]  # (T, M, 3), finite
    template_frame = int(persistent["template_frame"])
    template_node = persistent["template_node"]  # (M, 3)
    T, M, _ = trajectory.shape

    # --- Step 2: FPS candidates at the template frame ---
    candidates = fps_template(template_node, args.max_candidates)
    C = len(candidates)
    cand_traj = trajectory[:, candidates]  # (T, C, 3)
    cand_template = template_node[candidates].numpy()

    # --- Step 3: MST over mean-3D-distance-across-all-frames ---
    dist = (cand_traj[:, :, None] - cand_traj[:, None, :]).norm(dim=-1).mean(dim=0)  # (C, C)
    dist_np = dist.numpy()
    mst_sparse = minimum_spanning_tree(dist_np).tocoo()
    mst_edges = list(zip(mst_sparse.row.tolist(), mst_sparse.col.tolist()))
    mst_weights = mst_sparse.data.tolist()
    mean_edge_len = float(np.mean(mst_weights)) if mst_weights else 0.0

    adj: dict[int, set] = defaultdict(set)
    for i, j in mst_edges:
        adj[i].add(j)
        adj[j].add(i)
    for c in range(C):
        adj.setdefault(c, set())

    # --- Step 4: root = tree-center restricted to degree>=3 candidates ---
    degree0 = current_degree(adj)
    junctions0 = [n for n in adj if degree0[n] >= 3]
    root_fallback_used = len(junctions0) == 0
    search_pool = junctions0 if junctions0 else list(adj.keys())
    best_root, best_ecc = None, float("inf")
    for j in search_pool:
        ecc = max(tree_distances_from(j, adj, dist_np).values())
        if ecc < best_ecc:
            best_ecc, best_root = ecc, j
    root = best_root

    parent0, _order0 = bfs_parents(root, adj)

    # --- Step 5: prune short terminal branches (hop count AND cumulative length) ---
    leaves = [n for n in adj if degree0.get(n, 0) == 1 and n != root]
    pruned_nodes: set[int] = set()
    num_pruned_branches = 0
    for leaf in leaves:
        path = []
        hops = 0
        length = 0.0
        cur = leaf
        while True:
            path.append(cur)
            if degree0[cur] >= 3 or cur == root:
                break
            nxt = parent0[cur]
            length += float(dist_np[cur, nxt])
            hops += 1
            cur = nxt
        branch_nodes = path[:-1]
        if not branch_nodes:
            continue
        if hops <= args.max_terminal_hops and length <= args.max_terminal_hops * mean_edge_len:
            pruned_nodes.update(branch_nodes)
            num_pruned_branches += 1

    for n in pruned_nodes:
        for nb in list(adj[n]):
            adj[nb].discard(n)
        del adj[n]

    # --- Step 6: merge duplicate (nearby) junctions -- dynamic degree, distance-gated ---
    num_merges = 0
    while True:
        degree = current_degree(adj)
        pair = None
        for p in list(adj.keys()):
            if degree.get(p, 0) < 3:
                continue
            for c in list(adj[p]):
                if degree.get(c, 0) >= 3 and float(dist_np[p, c]) < mean_edge_len:
                    pair = (p, c)
                    break
            if pair:
                break
        if pair is None:
            break
        p, c = pair
        hops_from_root = bfs_parents(root, adj)[0]

        def _depth(n: int, parent_map=hops_from_root) -> int:
            depth = 0
            cur = n
            while parent_map.get(cur) is not None:
                cur = parent_map[cur]
                depth += 1
            return depth

        if _depth(p) > _depth(c):
            p, c = c, p
        for nb in list(adj[c]):
            if nb == p:
                continue
            adj[p].add(nb)
            adj[nb].discard(c)
            adj[nb].add(p)
        adj[p].discard(c)
        del adj[c]
        num_merges += 1

    # --- Step 7: simplify degree-2 chains (global worklist) ---
    threshold = args.simplify_dist_ratio * mean_edge_len
    degree = current_degree(adj)
    always_keep = {n for n in adj if degree[n] != 2} | {root}

    visited_edges: set[frozenset] = set()
    chains: list[tuple[int, int, list[int]]] = []
    for a in always_keep:
        for nb in list(adj[a]):
            e = frozenset((a, nb))
            if e in visited_edges:
                continue
            if degree[nb] != 2:
                visited_edges.add(e)
                chains.append((a, nb, []))
            else:
                end_anchor, interior = walk_chain(a, nb, adj, degree)
                path_nodes = [a] + interior + [end_anchor]
                for i in range(len(path_nodes) - 1):
                    visited_edges.add(frozenset((path_nodes[i], path_nodes[i + 1])))
                chains.append((a, end_anchor, interior))

    final_edges_local: list[tuple[int, int]] = []
    chain_stats = []
    for a, b, interior in chains:
        if not interior:
            final_edges_local.append((a, b))
            continue
        kept = simplify_chain(interior, a, b, cand_traj, threshold, args.max_inserts_per_chain)
        kept_sorted = sorted(kept, key=interior.index)
        path = [a] + kept_sorted + [b]
        for i in range(len(path) - 1):
            final_edges_local.append((path[i], path[i + 1]))
        chain_stats.append({"chain_length": len(interior), "kept": len(kept_sorted)})

    # --- Step 8: one fresh BFS from root over the final graph ---
    final_adj: dict[int, set] = defaultdict(set)
    for i, j in final_edges_local:
        final_adj[i].add(j)
        final_adj[j].add(i)
    parent_of, bfs_order = bfs_parents(root, final_adj)

    J = len(bfs_order)
    remap = {local: idx for idx, local in enumerate(bfs_order)}
    parents_arr = torch.full((J,), -1, dtype=torch.long)
    for local in bfs_order:
        p = parent_of[local]
        if p is not None:
            parents_arr[remap[local]] = remap[p]
    edges_arr = torch.tensor(
        [[i, int(parents_arr[i])] for i in range(J) if int(parents_arr[i]) != -1], dtype=torch.long
    )
    persistent_node_indices = torch.tensor([candidates[local] for local in bfs_order], dtype=torch.long)
    joints_template = template_node[persistent_node_indices].clone()
    joint_trajectories = trajectory[:, persistent_node_indices].clone()

    # --- Verification ---
    edges_list = edges_arr.tolist()
    if edges_list:
        row = np.array([e[0] for e in edges_list] + [e[1] for e in edges_list])
        col = np.array([e[1] for e in edges_list] + [e[0] for e in edges_list])
        adj_sparse = np.zeros((J, J), dtype=bool)
        adj_sparse[row, col] = True
        n_components, _ = connected_components(adj_sparse, directed=False)
    else:
        n_components = 1 if J == 1 else J
    is_connected = n_components == 1
    is_tree = is_connected and (edges_arr.shape[0] == J - 1)
    root_count = int((parents_arr == -1).sum().item())
    finite_ok = bool(torch.isfinite(joint_trajectories).all().item())
    unique_ok = bool(torch.unique(persistent_node_indices).numel() == J)

    verification = {
        "connected": is_connected,
        "is_tree": is_tree,
        "edge_count_equals_J_minus_1": bool(edges_arr.shape[0] == J - 1),
        "single_root": root_count == 1,
        "finite_trajectory": finite_ok,
        "no_duplicate_indices": unique_ok,
    }

    # --- Save skeleton.pt ---
    thresholds = {
        "max_candidates": args.max_candidates, "max_terminal_hops": args.max_terminal_hops,
        "simplify_dist_ratio": args.simplify_dist_ratio, "max_inserts_per_chain": args.max_inserts_per_chain,
    }
    payload = {
        "joints_template": joints_template,
        "joint_trajectories": joint_trajectories,
        "parents": parents_arr,
        "edges": edges_arr,
        "persistent_node_indices": persistent_node_indices,
        "template_frame": template_frame,
        "thresholds": thresholds,
        "meta": {
            "seq_name": args.seq_name, "config_path": str(config_path),
            "persistent_nodes_path": str(persistent_nodes_path), "num_frames": T,
        },
    }
    torch.save(payload, skeleton_path)

    # --- OBJ exports ---
    write_obj(output_dir / "candidate_mst.obj", cand_template, mst_edges)
    write_obj(output_dir / "skeleton.obj", joints_template.numpy(), edges_list)

    # --- Previews + video ---
    proj = _project_2d(joint_trajectories, dataset.Ks, dataset.w2cs)  # (T, J, 2)
    preview_paths, video_path = save_previews_and_video(dataset, proj, edges_list, output_dir, fps=10)

    output_paths = {
        "output_dir": output_dir,
        "skeleton": skeleton_path,
        "candidate_mst_obj": output_dir / "candidate_mst.obj",
        "skeleton_obj": output_dir / "skeleton.obj",
        "skeleton_overlay_video": video_path,
        **{p.stem: p for p in preview_paths},
    }
    elapsed_seconds = time.time() - start_time

    report = {
        "seq_name": args.seq_name,
        "config_path": str(config_path),
        "persistent_nodes_path": str(persistent_nodes_path),
        "output_dir": str(output_dir),
        "num_frames": T,
        "thresholds": thresholds,
        "funnel": {
            "num_persistent_nodes": M,
            "num_candidates_after_fps": C,
            "num_mst_edges": len(mst_edges),
            "num_junctions_found": len(junctions0),
            "root_fallback_used": root_fallback_used,
            "root_candidate_local_index": root,
            "root_persistent_node_index": candidates[root],
            "num_terminal_branches_pruned": num_pruned_branches,
            "num_nodes_pruned": len(pruned_nodes),
            "num_junction_merges": num_merges,
            "num_chains": len(chains),
            "chain_simplification": chain_stats,
            "final_num_joints": J,
            "final_num_edges": int(edges_arr.shape[0]),
        },
        "symmetry_correction": {
            "enabled": False,
            "reason": "no semantic left/right side labels available in this pipeline",
        },
        "verification": verification,
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "elapsed_seconds": elapsed_seconds,
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2)

    print(
        f"[extract_riggs_skeleton] seq={args.seq_name} candidates={C} mst_edges={len(mst_edges)} "
        f"pruned_branches={num_pruned_branches} merges={num_merges} final_joints={J} "
        f"verification={verification} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
