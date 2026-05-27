"""This file implements an inference pipeline for Point-GNN on KITTI dataset"""

import os
import time
import argparse
import csv
import numpy as np
from util.tf_compat import tf
import cv2
from tqdm import tqdm
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from collections import Counter

from util import open3d_compat as open3d
from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
                          get_encoding_len
from models import nms
from util.config_util import load_config

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str,
                    help='Path to checkpoint')
parser.add_argument('-l', '--level', type=int, default=0,
                    help='Visualization level: 0=off, 1=nonblocking, 2=block')
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True')
parser.add_argument('--dataset_root_dir', type=str,
                    default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--max_frames', type=int, default=None,
                    help='Max frames to run. None = full split.')

args = parser.parse_args()

MAX_FRAMES         = args.max_frames
VISUALIZATION_LEVEL = args.level
IS_TEST            = args.test
USE_BOX_MERGE      = args.use_box_merge
USE_BOX_SCORE      = args.use_box_score
DATASET_DIR        = args.dataset_root_dir

if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR, './3DOP_splits/val.txt')
else:
    DATASET_SPLIT_FILE = args.dataset_split_file

if args.output_dir == '':
    OUTPUT_DIR = os.path.join(args.checkpoint_path, './eval/')
else:
    OUTPUT_DIR = args.output_dir

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH     = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), 'No config file found in %s'
config = load_config(CONFIG_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '',
        num_classes=config['num_classes'],
        is_training=False)
else:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])

NUM_TEST_SAMPLE = dataset.num_files
NUM_CLASSES     = dataset.num_classes

# ── Occlusion helper ──────────────────────────────────────────────────────────
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected    = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:,0])-np.min(projected[:,0]))/(upper[0]-lower[0])
    y_cover_rate = (np.max(projected[:,1])-np.min(projected[:,1]))/(upper[1]-lower[1])
    z_cover_rate = (np.max(projected[:,2])-np.min(projected[:,2]))/(upper[2]-lower[2])
    return x_cover_rate * y_cover_rate * z_cover_rate

# ── Model setup ───────────────────────────────────────────────────────────────
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn  = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] == 'irgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
elif config['input_features'] == '0000':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i000':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])
elif config['input_features'] == '0':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])

t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.placeholder(dtype=tf.float32, shape=[None, 3]))

t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(tf.placeholder(dtype=tf.int32, shape=[None, 2]))

t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 1]))

t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(
    num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN,
    mode='test',
    **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list,
    t_keypoint_indices_list, t_edges_list, t_is_training)

t_probs       = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
global_step   = tf.Variable(0, dtype=tf.int32, trainable=False)

fetches = {
    'step':        global_step,
    'predictions': t_predictions,
    'probs':       t_probs,
    'pred_box':    t_pred_box,
}

# ── Visualization setup (level 1 only) ───────────────────────────────────────
color_map = np.array([
    (211,211,211),(255,0,0),(255,20,147),(65,244,101),
    (169,244,65),(65,79,244),(65,181,244),(229,244,66)],
    dtype=np.float32) / 255.0

gt_color_map = {
    'Pedestrian':    (0,255,255),
    'Person_sitting':(218,112,214),
    'Car':           (154,205,50),
    'Truck':         (255,215,0),
    'Van':           (255,20,147),
    'Tram':          (250,128,114),
    'Misc':          (128,0,128),
    'Cyclist':       (255,165,0),
}

if VISUALIZATION_LEVEL == 1:
    print("Configure the viewpoint as you want and press [q]")
    calib = dataset.get_calib(0)
    cam_points_in_img_with_rgb = dataset.get_cam_points_in_image_with_rgb(
        0, calib=calib)
    vis = open3d.Visualizer()
    vis.create_window()
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(cam_points_in_img_with_rgb.xyz)
    pcd.colors = open3d.Vector3dVector(
        cam_points_in_img_with_rgb.attr[:,1:4])
    line_set       = open3d.LineSet()
    graph_line_set = open3d.LineSet()
    box_corners = np.array([[0,0,0]])
    box_edges   = np.array([[0,0]])
    line_set.points       = open3d.Vector3dVector(box_corners)
    line_set.lines        = open3d.Vector2iVector(box_edges)
    graph_line_set.points = open3d.Vector3dVector(box_corners)
    graph_line_set.lines  = open3d.Vector2iVector(box_edges)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(graph_line_set)
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 3141.0, 0)
    vis.run()

# ── Session ───────────────────────────────────────────────────────────────────
time_dict   = {}
saver       = tf.train.Saver()
graph       = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

num_graph_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])

with tf.Session(graph=graph,
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    # Save TensorBoard graph
    writer = tf.summary.FileWriter('./graph_logs', sess.graph)
    writer.close()
    print("Graph saved to ./graph_logs")

    # ── Operation graph analysis ──────────────────────────────────────────────
    operations = graph.get_operations()
    op_types   = Counter(op.type for op in operations)

    print(f"\n{'='*65}")
    print(f"  OPERATION GRAPH ANALYSIS")
    print(f"{'='*65}")
    print(f"  Total operations in graph: {len(operations)}")

    print(f"\n  TOP OPERATION TYPES (by call count):")
    print(f"  {'Op Type':<30} {'Count':>6}")
    print(f"  {'-'*40}")
    for op_type, count in sorted(op_types.items(), key=lambda x: -x[1]):
        print(f"  {op_type:<30} {count:>6}")

    print(f"\n  GRAPH-SPECIFIC OPERATIONS PER LAYER:")
    print(f"  (GatherV2, UnsortedSegmentMax, ConcatV2, Sub)")
    print(f"  {'-'*60}")
    graph_ops = ['GatherV2', 'UnsortedSegmentMax', 'ConcatV2', 'Sub']
    for layer in ['layer1', 'layer2', 'layer3', 'layer4', 'output']:
        layer_ops = [op for op in operations if layer in op.name]
        if layer_ops:
            print(f"\n  {layer}:")
            for target_op in graph_ops:
                matching = [op.name for op in layer_ops
                            if op.type == target_op]
                for name in matching:
                    print(f"    {target_op:<25} → {name}")

    print(f"\n  MLP STRUCTURE PER LAYER (MatMul → BiasAdd → ReLU):")
    print(f"  {'-'*60}")
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        matmuls = [op.name for op in operations
                   if op.type == 'MatMul' and layer in op.name]
        relus   = [op.name for op in operations
                   if op.type == 'Relu'   and layer in op.name]
        print(f"\n  {layer}: {len(matmuls)} MatMul ops, {len(relus)} ReLU ops")
        for name in relus:
            print(f"    ReLU → {name}")

    print(f"\n  KEY TENSOR SHAPES:")
    print(f"  {'-'*60}")
    for op in operations:
        if op.type in ['GatherV2', 'UnsortedSegmentMax', 'ConcatV2']:
            try:
                shape = op.outputs[0].shape
                print(f"  {op.type:<25} output shape: {shape}  → {op.name}")
            except:
                pass
    print(f"\n{'='*65}\n")

    # ── Restore checkpoint ────────────────────────────────────────────────────
    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restore from checkpoint: {model_path}')
    saver.restore(sess, model_path)

    frames_to_run = NUM_TEST_SAMPLE
    if MAX_FRAMES is not None:
        frames_to_run = min(NUM_TEST_SAMPLE, MAX_FRAMES)

    coo_analysis_done = False   # run COO analysis on first frame only

    # ── Main loop ─────────────────────────────────────────────────────────────
    for frame_idx in tqdm(range(frames_to_run)):

        start_time = time.time()

        if VISUALIZATION_LEVEL == 2:
            pcd            = open3d.PointCloud()
            line_set       = open3d.LineSet()
            graph_line_set = open3d.LineSet()

        # Input
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        calib = dataset.get_calib(frame_idx)
        image = dataset.get_image(frame_idx)
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        input_time = time.time()
        time_dict['fetch input'] = (time_dict.get('fetch input', 0)
                                    + input_time - start_time)

        # Graph construction
        graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list,
         keypoint_indices_list,
         edges_list) = graph_generate_fn(
            cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
        graph_time = time.time()
        time_dict['gen graph'] = (time_dict.get('gen graph', 0)
                                  + graph_time - input_time)

        # ── Per-frame graph statistics ────────────────────────────────────────
        lvl0_edges = lvl0_verts = lvl0_avg = lvl0_min = lvl0_max = 0
        lvl1_edges = lvl1_verts = lvl1_avg = lvl1_min = lvl1_max = 0

        for lvl, edges in enumerate(edges_list):
            num_edges  = edges.shape[0]
            src_counts = np.bincount(
                edges[:, 0],
                minlength=vertex_coord_list[lvl].shape[0])
            mask      = src_counts > 0
            num_src_v = int(mask.sum())
            if num_src_v:
                avg_nei = float(src_counts[mask].mean())
                min_nei = int(src_counts[mask].min())
                max_nei = int(src_counts[mask].max())
            else:
                avg_nei = min_nei = max_nei = 0
            locals()[f'lvl{lvl}_edges'] = num_edges
            locals()[f'lvl{lvl}_verts'] = num_src_v
            locals()[f'lvl{lvl}_avg']   = avg_nei
            locals()[f'lvl{lvl}_min']   = min_nei
            locals()[f'lvl{lvl}_max']   = max_nei

        NUM_LEVELS    = len(edges_list)
        gather_calls  = 3 * (lvl0_edges + lvl1_edges)
        concat_calls  = 1 * (lvl0_edges + lvl1_edges)
        scatter_calls = 1 * (lvl0_edges + lvl1_edges)
        matmul_calls  = 6 * NUM_LEVELS
        biasadd_calls = 6 * NUM_LEVELS
        relu_calls    = 5 * NUM_LEVELS

        print(
            f"Frame {frame_idx:06d} | "
            f"L0: E={lvl0_edges:7d} V={lvl0_verts:5d} "
            f"⟨k⟩={lvl0_avg:5.1f}  min={lvl0_min:2d}  max={lvl0_max:3d} | "
            f"L1: E={lvl1_edges:7d} V={lvl1_verts:5d} "
            f"⟨k⟩={lvl1_avg:5.1f}  min={lvl1_min:2d}  max={lvl1_max:3d} | "
            f"GatherV2:{gather_calls:9d}  ConcatV2:{concat_calls:8d}  "
            f"UnsortedSegMax:{scatter_calls:9d}  "
            f"MatMul:{matmul_calls:3d}  BiasAdd:{biasadd_calls:3d}  "
            f"ReLU:{relu_calls:3d}"
        )

        # ── COO Sparsity Analysis (frame 0 only) ──────────────────────────────
        if not coo_analysis_done:
            coo_analysis_done = True

            print(f"\n{'='*65}")
            print(f"  COO FORMAT SPARSITY ANALYSIS — Frame {frame_idx}")
            print(f"{'='*65}")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                'Point-GNN Graph — COO Format Sparsity Pattern',
                fontsize=14, fontweight='bold')

            for lvl, edges in enumerate(edges_list):
                if lvl >= 2:
                    break

                num_v    = vertex_coord_list[lvl].shape[0]
                src      = edges[:, 0]
                dst      = edges[:, 1]
                num_e    = edges.shape[0]
                max_v    = max(num_v,
                               int(src.max()) + 1,
                               int(dst.max()) + 1)
                possible = max_v * max_v
                sparsity = 1.0 - num_e / possible

                print(f"\n  Graph Level {lvl}:")
                print(f"    Vertices:           {max_v:>10,}")
                print(f"    Actual edges:       {num_e:>10,}")
                print(f"    Possible edges:     {possible:>10,}")
                print(f"    Sparsity:           {sparsity:>10.4%}")
                print(f"    COO storage:        "
                      f"{num_e*2*4/1024:.1f} KB  "
                      f"({num_e} rows × 2 cols × 4 bytes)")
                print(f"    Dense adj matrix:   "
                      f"{possible*4/1024/1024:.1f} MB  "
                      f"(would need {max_v}×{max_v})")
                print(f"    Memory saving:      "
                      f"{possible*4/(num_e*2*4):.1f}x "
                      f"vs dense adjacency matrix")

                # Subsample for spy plot
                max_plot = 5000
                if num_e > max_plot:
                    idx     = np.random.choice(num_e, max_plot, replace=False)
                    src_plt = src[idx]
                    dst_plt = dst[idx]
                    suffix  = f'\n(showing {max_plot:,} of {num_e:,} edges)'
                else:
                    src_plt = src
                    dst_plt = dst
                    suffix  = f'\n({num_e:,} edges total)'

                data   = np.ones(len(src_plt), dtype=np.float32)
                sparse = coo_matrix(
                    (data, (src_plt, dst_plt)),
                    shape=(max_v, max_v))

                ax = axes[lvl]
                ax.spy(sparse, markersize=0.3,
                       color='#2196F3', alpha=0.6)
                ax.set_title(
                    f'Level {lvl} Adjacency — COO Format{suffix}\n'
                    f'Sparsity: {sparsity:.2%} | '
                    f'{max_v} × {max_v} vertex space',
                    fontsize=9, fontweight='bold')
                ax.set_xlabel('Destination vertex index', fontsize=8)
                ax.set_ylabel('Source vertex index',      fontsize=8)
                ax.tick_params(labelsize=7)

            # Third panel — degree distribution
            ax3 = axes[2]
            level_colors = ['#FF8C00', '#4CAF50']
            for lvl, edges in enumerate(edges_list):
                if lvl >= 2:
                    break
                num_v  = vertex_coord_list[lvl].shape[0]
                counts = np.bincount(edges[:, 0], minlength=num_v)
                mask   = counts > 0
                c      = counts[mask]
                ax3.hist(c, bins=60,
                         alpha=0.6,
                         color=level_colors[lvl],
                         label=f'Level {lvl}  '
                               f'(min={c.min()}, '
                               f'max={c.max()}, '
                               f'mean={c.mean():.1f})',
                         density=True,
                         edgecolor='none')
                ax3.axvline(x=c.min(), color=level_colors[lvl],
                            linestyle='--', linewidth=1.2, alpha=0.8)
                ax3.axvline(x=c.max(), color=level_colors[lvl],
                            linestyle=':',  linewidth=1.2, alpha=0.8)

            ax3.set_xlabel('Degree (neighbors per vertex)', fontsize=10)
            ax3.set_ylabel('Density',                       fontsize=10)
            ax3.set_title(
                'Degree Distribution\n'
                'Variable neighborhoods → scatter_max thread divergence',
                fontsize=9, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            plt.tight_layout()
            spy_path = os.path.join(OUTPUT_DIR, 'coo_sparsity_pattern.png')
            plt.savefig(spy_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"\n  COO sparsity plot saved → {spy_path}")
            print(f"{'='*65}\n")

        # ── Feature preparation ───────────────────────────────────────────────
        if config['input_features'] == 'irgb':
            input_v = cam_rgb_points.attr
        elif config['input_features'] == '0rgb':
            input_v = np.hstack([
                np.zeros((cam_rgb_points.attr.shape[0], 1)),
                cam_rgb_points.attr[:, 1:]])
        elif config['input_features'] == '0000':
            input_v = np.zeros_like(cam_rgb_points.attr)
        elif config['input_features'] == 'i000':
            input_v = np.hstack([
                cam_rgb_points.attr[:, [0]],
                np.zeros((cam_rgb_points.attr.shape[0], 3))])
        elif config['input_features'] == 'i':
            input_v = cam_rgb_points.attr[:, [0]]
        elif config['input_features'] == '0':
            input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

        last_layer_graph_level = \
            config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz  = \
            vertex_coord_list[last_layer_graph_level + 1]

        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1,
                         'Pedestrian': 3, 'Cyclist': 5, 'DontCare': 7}
        elif config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        elif config['label_method'] == 'Pedestrian_and_Cyclist':
            label_map = {'Background': 0, 'Pedestrian': 1,
                         'Cyclist': 3, 'DontCare': 5}

        feed_dict = {
            t_initial_vertex_features: input_v,
            t_is_training: False,
        }
        feed_dict.update(dict(zip(t_edges_list, edges_list)))
        feed_dict.update(
            dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))

        # ── GNN forward pass ─────────────────────────────────────────────────
        results  = sess.run(fetches, feed_dict=feed_dict)
        gnn_time = time.time()
        time_dict['gnn inference'] = (time_dict.get('gnn inference', 0)
                                      + gnn_time - graph_time)

        # ── Box decoding ──────────────────────────────────────────────────────
        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1))
        box_labels     = box_labels.reshape((-1))
        raw_box_labels = box_labels.copy()
        box_probs      = box_probs.reshape((-1))
        pred_boxes     = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))

        last_layer_points_xyz = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape((-1, 3))
        boxes_centers = last_layer_points_xyz

        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map)

        box_mask    = (box_labels > 0) * (box_labels < NUM_CLASSES - 1)
        box_mask    = box_mask * (box_probs > 1. / NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]
        decode_time = time.time()
        time_dict['decode box'] = (time_dict.get('decode box', 0)
                                   + decode_time - gnn_time)

        # ── NMS + output ──────────────────────────────────────────────────────
        pred_labels = []
        if box_indices.size != 0:
            box_labels_f    = box_labels[box_indices].copy()
            box_probs_f     = box_probs[box_indices].copy()
            box_probs_ori   = box_probs_f.copy()
            decoded_boxes_f = decoded_boxes[box_indices, 0]
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5
            detection_scores = box_probs_f

            if USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_uncertainty(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            elif USE_BOX_MERGE and not USE_BOX_SCORE:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_merge_only(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            elif not USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_score_only(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            else:
                (class_labels, detection_boxes_3d,
                 detection_scores, nms_indices) = \
                    nms.nms_boxes_3d(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))

            box_probs_f = detection_scores

            # Visualization
            if VISUALIZATION_LEVEL > 0:
                last_layer_points_color = np.zeros(
                    (last_layer_points_xyz.shape[0], 3), dtype=np.float32)
                last_layer_points_color[:, :] = color_map[raw_box_labels, :]
                cam_points_color = cam_rgb_points.attr[:, 1:]
                pcd.points = open3d.Vector3dVector(np.vstack([
                    last_layer_points_xyz[box_indices][nms_indices],
                    last_layer_points_xyz,
                    cam_rgb_points.xyz]))
                pcd.colors = open3d.Vector3dVector(np.vstack([
                    last_layer_points_color[box_indices][nms_indices],
                    np.tile([(1, 0., 200./255)],
                            (last_layer_points_color.shape[0], 1)),
                    cam_points_color]))
                output_points_idx = box_indices[nms_indices] // NUM_CLASSES
                edge_mask = np.isin(
                    edges_list[last_layer_graph_level][:, 1],
                    output_points_idx)
                last_layer_edges = np.hstack([
                    edges_list[last_layer_graph_level][:, [0]][edge_mask],
                    keypoint_indices_list[-1][
                        edges_list[last_layer_graph_level][:, 1][edge_mask]]])
                colors = last_layer_points_color[
                    edges_list[last_layer_graph_level][:, 1][edge_mask]]
                for i in range(len(keypoint_indices_list) - 2, -1, -1):
                    last_layer_edges = \
                        keypoint_indices_list[i][last_layer_edges, 0]
                last_layer_edges += len(box_indices[nms_indices])
                last_layer_edges += last_layer_points_xyz.shape[0]
                graph_line_set.points = pcd.points
                graph_line_set.lines  = open3d.Vector2iVector(
                    last_layer_edges)
                graph_line_set.colors = open3d.Vector3dVector(colors)

            # Convert to KITTI format
            detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                detection_boxes_3d)

            for i in range(len(detection_boxes_3d_corners)):
                corners_cam   = Points(
                    xyz=detection_boxes_3d_corners[i], attr=None)
                corners_img   = dataset.cam_points_to_image(
                    corners_cam, calib)
                corners_xy    = corners_img.xyz[:, :2]

                if config['label_method'] == 'yaw':
                    all_class_name = ['Background','Car','Car',
                                      'Pedestrian','Pedestrian',
                                      'Cyclist','Cyclist','DontCare']
                elif config['label_method'] == 'Car':
                    all_class_name = ['Background','Car','Car','DontCare']
                elif config['label_method'] == 'Pedestrian_and_Cyclist':
                    all_class_name = ['Background','Pedestrian','Pedestrian',
                                      'Cyclist','Cyclist','DontCare']
                else:
                    all_class_name = ['Background','Car','Car',
                                      'Pedestrian','Pedestrian',
                                      'Cyclist','Cyclist','DontCare']

                class_name = all_class_name[class_labels[i]]
                xmin, ymin = np.amin(corners_xy, axis=0)
                xmax, ymax = np.amax(corners_xy, axis=0)
                clip_xmin  = max(xmin, 0.0)
                clip_ymin  = max(ymin, 0.0)
                clip_xmax  = min(xmax, 1242.0)
                clip_ymax  = min(ymax, 375.0)
                trunc = 1.0 - ((clip_ymax-clip_ymin)*(clip_xmax-clip_xmin)
                               / ((ymax-ymin)*(xmax-xmin)))
                if trunc > 0.4:
                    continue

                x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                assert l > 0, str(i)
                score = box_probs_f[i]

                if USE_BOX_SCORE:
                    tmp_label = {'x3d': x3d, 'y3d': y3d, 'z3d': z3d,
                                 'yaw': yaw, 'height': h,
                                 'width': w, 'length': l}
                    inside_mask   = dataset.sel_xyz_in_box3d(
                        tmp_label,
                        last_layer_points_xyz[box_indices])
                    points_inside = \
                        last_layer_points_xyz[box_indices][inside_mask]
                    score = (1 + occlusion(tmp_label, points_inside)) * score

                pred_labels.append((
                    class_name, -1, -1, 0,
                    clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                    h, w, l, x3d, y3d, z3d, yaw, score))

                if VISUALIZATION_LEVEL > 0:
                    cv2.rectangle(image,
                        (int(clip_xmin), int(clip_ymin)),
                        (int(clip_xmax), int(clip_ymax)),
                        (0, 255, 0), 2)
                    label_str = 'P' if class_name == 'Pedestrian' else 'C'
                    cv2.putText(image,
                        f'{label_str} | {score:.3f}',
                        (int(clip_xmin), int(clip_ymin)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0) if class_name == 'Pedestrian'
                        else (0, 255, 0), 1)

            nms_time = time.time()
            time_dict['nms'] = (time_dict.get('nms', 0)
                                + nms_time - decode_time)

            # Write output
            filename = os.path.join(
                OUTPUT_DIR, 'data',
                dataset.get_filename(frame_idx) + '.txt')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                for pred_label in pred_labels:
                    for field in pred_label:
                        f.write(str(field) + ' ')
                    f.write('\n')
                f.write('\n')

            if VISUALIZATION_LEVEL > 0 and not IS_TEST:
                gt_boxes  = []
                gt_colors = []
                for label in box_label_list:
                    if label['name'] in gt_color_map:
                        gt_boxes.append([
                            label['x3d'], label['y3d'], label['z3d'],
                            label['length'], label['height'],
                            label['width'], label['yaw']])
                        gt_colors.append(gt_color_map[label['name']])
                gt_boxes  = np.array(gt_boxes)
                gt_colors = np.array(gt_colors) / 255.
                gt_box_corners, gt_box_edges, gt_box_colors = \
                    dataset.boxes_3d_to_line_set(
                        gt_boxes, boxes_color=gt_colors)
                if gt_box_corners is None or gt_box_corners.size < 1:
                    gt_box_corners = np.array([[0, 0, 0]])
                    gt_box_edges   = np.array([[0, 0]])
                    gt_box_colors  = np.array([[0, 0, 0]])
                box_corners, box_edges, box_colors = \
                    dataset.boxes_3d_to_line_set(detection_boxes_3d)

        else:
            # No detections this frame
            filename = os.path.join(
                OUTPUT_DIR, 'data',
                dataset.get_filename(frame_idx) + '.txt')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write('\n')

            if VISUALIZATION_LEVEL > 0:
                last_layer_points_color = np.zeros(
                    (last_layer_points_xyz.shape[0], 3), dtype=np.float32)
                last_layer_points_color[:, :] = color_map[raw_box_labels, :]
                cam_points_color = cam_rgb_points.attr[:, 1:]
                box_corners = np.array([[0, 0, 0]])
                box_edges   = np.array([[0, 0]])
                box_colors  = np.array([[0, 0, 0]])
                pcd.points  = open3d.Vector3dVector(np.vstack([
                    last_layer_points_xyz, cam_rgb_points.xyz]))
                pcd.colors  = open3d.Vector3dVector(np.vstack([
                    np.tile([(128./255, 0., 128./255)],
                            (last_layer_points_color.shape[0], 1)),
                    cam_points_color]))
                graph_line_set.points = open3d.Vector3dVector(
                    np.array([[0, 0, 0]]))
                graph_line_set.lines  = open3d.Vector2iVector([[0, 0]])
                graph_line_set.colors = open3d.Vector3dVector(
                    np.array([[0, 0, 0]]))

                if not IS_TEST:
                    gt_boxes  = []
                    gt_colors = []
                    for label in box_label_list:
                        if label['name'] in gt_color_map:
                            gt_boxes.append([
                                label['x3d'], label['y3d'], label['z3d'],
                                label['length'], label['height'],
                                label['width'], label['yaw']])
                            gt_colors.append(gt_color_map[label['name']])
                    gt_boxes  = np.array(gt_boxes)
                    gt_colors = np.array(gt_colors) / 255.
                    gt_box_corners, gt_box_edges, gt_box_colors = \
                        dataset.boxes_3d_to_line_set(
                            gt_boxes, boxes_color=gt_colors)
                    if gt_box_corners is None or gt_box_corners.size < 1:
                        gt_box_corners = np.array([[0, 0, 0]])
                        gt_box_edges   = np.array([[0, 0]])
                        gt_box_colors  = np.array([[0, 0, 0]])

        # ── Visualization update ──────────────────────────────────────────────
        if VISUALIZATION_LEVEL > 0:
            cv2.imshow('image', image)
            cv2.waitKey(10)

            if not IS_TEST and len(pred_labels) > 0:
                box_edges += gt_box_corners.shape[0]
                line_set.points = open3d.Vector3dVector(
                    np.vstack([gt_box_corners, box_corners]))
                line_set.lines  = open3d.Vector2iVector(
                    np.vstack([gt_box_edges, box_edges]))
                line_set.colors = open3d.Vector3dVector(
                    np.vstack([gt_box_colors, box_colors]))
            else:
                line_set.points = open3d.Vector3dVector(box_corners)
                line_set.lines  = open3d.Vector2iVector(box_edges)
                line_set.colors = open3d.Vector3dVector(box_colors)

        if VISUALIZATION_LEVEL == 1:
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

        if VISUALIZATION_LEVEL == 2:
            print("Configure the viewpoint and press [q]")
            def custom_draw(geometry_list):
                v = open3d.Visualizer()
                v.create_window()
                for g in geometry_list:
                    v.add_geometry(g)
                ctr = v.get_view_control()
                ctr.rotate(0.0, 3141.0, 0)
                v.run()
                v.destroy_window()
            custom_draw([pcd, line_set, graph_line_set])

        total_time = time.time()
        time_dict['total'] = (time_dict.get('total', 0)
                              + total_time - start_time)

    # ── Final timing summary ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TIMING SUMMARY OVER {frames_to_run} FRAMES")
    print(f"{'='*65}")
    for key in time_dict:
        print(f"  {key:<20} {time_dict[key]/frames_to_run:>8.1f}ms avg")
    print(f"{'='*65}\n")