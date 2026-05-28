"""The file defines functions to generate graphs."""

import time
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
from util import open3d_compat as open3d

# ── Sub-stage timing (written by gen_multi_level_local_graph_v3) ──────────────
_gc_timing = {
    'downsample_ms': 0.0,
    'edge_build_ms': 0.0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Downsampling helpers
# ══════════════════════════════════════════════════════════════════════════════

def multi_layer_downsampling(points_xyz, base_voxel_size,
                             levels=None, add_rnd3d=False):
    """Downsample the points using base_voxel_size at different scales."""
    if levels is None:
        levels = [1]

    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])

    downsampled_list = [points_xyz]
    last_level = 0

    for level in levels:
        if np.isclose(last_level, level):
            downsampled_list.append(np.copy(downsampled_list[-1]))
        else:
            if add_rnd3d:
                xyz_idx = (
                    points_xyz - xyz_offset
                    + base_voxel_size * level * np.random.random((1, 3))
                ) // (base_voxel_size * level)
                xyz_idx = xyz_idx.astype(np.int32)
                dim_x, dim_y, _ = np.amax(xyz_idx, axis=0) + 1
                keys = (xyz_idx[:, 0]
                        + xyz_idx[:, 1] * dim_x
                        + xyz_idx[:, 2] * dim_y * dim_x)
                sorted_order      = np.argsort(keys)
                sorted_keys       = keys[sorted_order]
                sorted_points_xyz = points_xyz[sorted_order]
                _, lens           = np.unique(sorted_keys, return_counts=True)
                indices           = np.hstack([[0], lens[:-1]]).cumsum()
                downsampled_xyz   = (
                    np.add.reduceat(sorted_points_xyz, indices, axis=0)
                    / lens[:, np.newaxis])
                downsampled_list.append(np.array(downsampled_xyz))
            else:
                pcd        = open3d.PointCloud()
                pcd.points = open3d.Vector3dVector(points_xyz)
                downsampled_xyz = np.asarray(
                    open3d.voxel_down_sample(
                        pcd, voxel_size=base_voxel_size * level).points)
                downsampled_list.append(downsampled_xyz)
        last_level = level

    return downsampled_list


def multi_layer_downsampling_select(points_xyz, base_voxel_size,
                                    levels=None, add_rnd3d=False):
    """Downsample at different scales; match downsampled pts to originals
    via nearest-neighbour search.

    Returns: vertex_coord_list, keypoint_indices_list
    """
    if levels is None:
        levels = [1]

    vertex_coord_list = multi_layer_downsampling(
        points_xyz, base_voxel_size,
        levels=levels, add_rnd3d=add_rnd3d)

    num_levels = len(vertex_coord_list)
    assert num_levels == len(levels) + 1

    keypoint_indices_list = []
    last_level = 0

    for i in range(1, num_levels):
        current_level = levels[i - 1]
        base_points   = vertex_coord_list[i - 1]
        current_points= vertex_coord_list[i]

        if np.isclose(current_level, last_level):
            # Same scale — identity mapping, no search needed
            vertex_coord_list[i] = base_points
            keypoint_indices_list.append(
                np.expand_dims(np.arange(base_points.shape[0]), axis=1))
        else:
            # Different scale — KD-tree nearest-neighbour search
            nbrs    = NearestNeighbors(
                n_neighbors=1, algorithm='kd_tree', n_jobs=1).fit(base_points)
            indices = nbrs.kneighbors(current_points, return_distance=False)
            vertex_coord_list[i] = base_points[indices[:, 0], :]
            keypoint_indices_list.append(indices)

        last_level = current_level

    return vertex_coord_list, keypoint_indices_list


def multi_layer_downsampling_random(points_xyz, base_voxel_size,
                                    levels=None, add_rnd3d=False):
    """Downsample at different scales by randomly selecting one point
    per voxel cell.

    Returns: vertex_coord_list, keypoint_indices_list
    """
    if levels is None:
        levels = [1]

    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])

    vertex_coord_list     = [points_xyz]
    keypoint_indices_list = []
    last_level            = 0

    for level in levels:
        last_points_xyz = vertex_coord_list[-1]

        if np.isclose(last_level, level):
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                          // (base_voxel_size * level)
            else:
                xyz_idx = (
                    last_points_xyz - xyz_offset
                    + base_voxel_size * level * np.random.random((1, 3))
                ) // (base_voxel_size * level)

            xyz_idx        = xyz_idx.astype(np.int32)
            dim_x, dim_y, _= np.amax(xyz_idx, axis=0) + 1
            keys           = (xyz_idx[:, 0]
                              + xyz_idx[:, 1] * dim_x
                              + xyz_idx[:, 2] * dim_y * dim_x)

            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            downsampled_xyz     = []
            downsampled_xyz_idx = []
            for key in voxels_idx:
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)

            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx), axis=1))

        last_level = level

    return vertex_coord_list, keypoint_indices_list


# ══════════════════════════════════════════════════════════════════════════════
#  Edge / graph generation (leaf-level, no recursion)
# ══════════════════════════════════════════════════════════════════════════════

def gen_disjointed_rnn_local_graph_v3(
        points_xyz, center_xyz, radius, num_neighbors,
        neighbors_downsample_method='random',
        scale=None):
    """Generate a local graph by radius-neighbour search.

    This is a *leaf* graph-gen function — it does NOT call
    get_graph_generate_fn, so NearestNeighbors is always in scope here.
    """
    if scale is not None:
        scale      = np.array(scale)
        points_xyz = points_xyz / scale
        center_xyz = center_xyz / scale

    nbrs    = NearestNeighbors(
        radius=radius, algorithm='ball_tree', n_jobs=1).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)

    if num_neighbors > 0 and neighbors_downsample_method == 'random':
        indices = [
            neighbors if neighbors.size <= num_neighbors
            else np.random.choice(neighbors, num_neighbors, replace=False)
            for neighbors in indices
        ]

    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate([
        i * np.ones(neighbors.size, dtype=np.int32)
        for i, neighbors in enumerate(indices)
    ])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices


# ══════════════════════════════════════════════════════════════════════════════
#  Registry — defined BEFORE gen_multi_level_local_graph_v3 so the forward
#  reference inside that function always resolves correctly.
# ══════════════════════════════════════════════════════════════════════════════

def get_graph_generate_fn(method_name):
    method_map = {
        'disjointed_rnn_local_graph_v3': gen_disjointed_rnn_local_graph_v3,
        'multi_level_local_graph_v3':    gen_multi_level_local_graph_v3,
    }
    if method_name not in method_map:
        raise ValueError(
            f"Unknown graph gen method '{method_name}'. "
            f"Available: {list(method_map.keys())}")
    return method_map[method_name]


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level multi-scale graph builder — calls get_graph_generate_fn safely
#  because the registry is already defined above.
# ══════════════════════════════════════════════════════════════════════════════

def gen_multi_level_local_graph_v3(
        points_xyz, base_voxel_size, level_configs,
        add_rnd3d=False, downsample_method='center'):
    """Generate graphs at multiple scales.

    Enforces that the output vertices of graph i match the input vertices
    of graph i+1 so GNN layers can be applied sequentially.

    Args:
        points_xyz:       [N, 3] float array of point coordinates.
        base_voxel_size:  scalar (or list) voxel cell size.
        level_configs:    list of dicts with keys:
                            'graph_scale', 'graph_level',
                            'graph_gen_method', 'graph_gen_kwargs'.
        add_rnd3d:        add random jitter during voxel downsampling.
        downsample_method: 'center' (default) or 'random'.

    Returns:
        vertex_coord_list, keypoint_indices_list, edges_list
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)

    scales = [cfg['graph_scale'] for cfg in level_configs]

    # ── Sub-stage 1: Voxel downsample ─────────────────────────────────────
    t_ds0 = time.perf_counter()
    if downsample_method == 'center':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_select(
                points_xyz, base_voxel_size,
                levels=scales, add_rnd3d=add_rnd3d)
    elif downsample_method == 'random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size,
                levels=scales, add_rnd3d=add_rnd3d)
    else:
        raise ValueError(
            f"Unknown downsample_method '{downsample_method}'. "
            "Choose 'center' or 'random'.")
    t_ds1 = time.perf_counter()
    _gc_timing['downsample_ms'] = (t_ds1 - t_ds0) * 1000

    # ── Sub-stage 2: Edge construction (FRNN + edge list build) ───────────
    t_eb0 = time.perf_counter()
    edges_list = []
    for cfg in level_configs:
        graph_level  = cfg['graph_level']
        gen_graph_fn = get_graph_generate_fn(cfg['graph_gen_method'])
        src_xyz      = vertex_coord_list[graph_level]
        center_xyz   = vertex_coord_list[graph_level + 1]
        edges        = gen_graph_fn(src_xyz, center_xyz,
                                    **cfg['graph_gen_kwargs'])
        edges_list.append(edges)
    t_eb1 = time.perf_counter()
    _gc_timing['edge_build_ms'] = (t_eb1 - t_eb0) * 1000

    return vertex_coord_list, keypoint_indices_list, edges_list