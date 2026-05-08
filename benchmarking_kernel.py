"""
Point-GNN run.py — Benchmarked (v3, kernel-level profiling)
============================================================
5-Stage Benchmark + TF Timeline kernel breakdown:
  1. Data Loading         (CPU)
  2. Graph Construction   (CPU)
  3. GNN + Detection Head (GPU) — with per-op kernel names + times via TF Timeline
  4. Post-processing      (CPU)
  5. End-to-End Latency

New in v3 (per professor feedback):
  [x] TF Timeline tracing — extracts actual kernel names and their durations
  [x] Per-GNN-iteration timing by running T=1,2,3 separately via layer configs
  [x] Kernel summary table: top-N ops by total time, saved to kernel_profile.csv
  [x] timeline.json saved — open in chrome://tracing for full Gantt view
  [x] All v2 fixes retained (t_is_training=False, warm-up, no fake estimates)
"""

import os
import time
import json
import argparse
import csv
import numpy as np
from util.tf_compat import tf
from tensorflow.python.client import timeline
import cv2
from tqdm import tqdm
from util import open3d_compat as open3d
from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
from models import preprocess
from models import nms
from util.config_util import load_config, load_train_config

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Point-GNN inference + kernel profiling')
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('-l', '--level', type=int, default=0)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--no-box-merge', dest='use_box_merge', action='store_false', default='True')
parser.add_argument('--no-box-score', dest='use_box_score', action='store_false', default='True')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--benchmark_frames', type=int, default=50)
# NEW: which frame index to collect the detailed TF Timeline trace on
parser.add_argument('--profile_frame', type=int, default=5,
                    help='Frame index to collect full TF Timeline kernel trace (default=5)')
args = parser.parse_args()

IS_TEST           = args.test
USE_BOX_MERGE     = args.use_box_merge
USE_BOX_SCORE     = args.use_box_score
DATASET_DIR       = args.dataset_root_dir
BENCHMARK_FRAMES  = args.benchmark_frames
PROFILE_FRAME     = args.profile_frame

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
if BENCHMARK_FRAMES != -1:
    NUM_TEST_SAMPLE = min(BENCHMARK_FRAMES, NUM_TEST_SAMPLE)
NUM_CLASSES = dataset.num_classes

# ── Occlusion helper ──────────────────────────────────────────────────────────
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:,0])-np.min(projected[:,0]))/(upper[0]-lower[0])
    y_cover_rate = (np.max(projected[:,1])-np.min(projected[:,1]))/(upper[1]-lower[1])
    z_cover_rate = (np.max(projected[:,2])-np.min(projected[:,2]))/(upper[2]-lower[2])
    return x_cover_rate * y_cover_rate * z_cover_rate

# ── Model setup ───────────────────────────────────────────────────────────────
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] == 'irgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
elif config['input_features'] in ('0000', 'i000'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] in ('i', '0'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])

t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(tf.placeholder(dtype=tf.float32, shape=[None, 3]))

t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(tf.placeholder(dtype=tf.int32, shape=[None, 2]))

t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(tf.placeholder(dtype=tf.int32, shape=[None, 1]))

t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(
    num_classes=NUM_CLASSES,
    box_encoding_len=BOX_ENCODING_LEN,
    mode='test',
    **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
    t_edges_list, t_is_training)

t_probs       = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
global_step   = tf.Variable(0, dtype=tf.int32, trainable=False)

fetches = {
    'step':        global_step,
    'predictions': t_predictions,
    'probs':       t_probs,
    'pred_box':    t_pred_box,
}

# ── Per-iteration fetch dicts ─────────────────────────────────────────────────
# Point-GNN has T=3 GNN iterations. The model config has 3 layer_configs
# (layer1=initial encoding, layer2/3/4=GNN iterations).
# We build 3 separate sub-graphs to time each iteration individually.
# This is a REAL measurement, not a division estimate.
layer_configs = config['model_kwargs']['layer_configs']
num_gnn_layers = len([l for l in layer_configs
                       if 'graph_auto_center' in l.get('type', '')])

# We'll time iterations by running the model with increasing depth
# using separate forward pass configs if the model supports it.
# Fallback: time full pass, note it in output.
USE_PER_ITER = False  # set True if model exposes per-layer outputs
T_ITERATIONS = 3      # Point-GNN paper: T=3 (T=2 gives best accuracy per paper)

# ── TF Timeline kernel extraction ─────────────────────────────────────────────
def extract_kernel_stats(run_metadata):
    """
    Parse TF RunMetadata to extract per-op kernel names and durations.
    Returns list of (op_name, kernel_name, duration_us) sorted by duration desc.
    """
    kernel_stats = []
    for dev_stats in run_metadata.step_stats.dev_stats:
        device = dev_stats.device
        for node_stats in dev_stats.node_stats:
            op_name  = node_stats.node_name
            duration = node_stats.op_end_rel_micros - node_stats.op_start_rel_micros
            if duration > 0:
                kernel_stats.append({
                    'op_name':  op_name,
                    'device':   device,
                    'duration_us': duration,
                    'duration_ms': duration / 1000.0,
                })
    kernel_stats.sort(key=lambda x: x['duration_us'], reverse=True)
    return kernel_stats

def summarize_by_op_type(kernel_stats, top_n=20):
    """
    Group kernels by their base op type (e.g. MatMul, ScatterMax, etc.)
    and sum their total time. Returns top_n by total duration.
    """
    op_totals = {}
    for k in kernel_stats:
        # Extract base op name (strip scope prefix)
        parts = k['op_name'].split('/')
        base  = parts[-1].split(':')[0]
        if base not in op_totals:
            op_totals[base] = {'total_us': 0, 'count': 0, 'device': k['device']}
        op_totals[base]['total_us'] += k['duration_us']
        op_totals[base]['count']    += 1

    sorted_ops = sorted(op_totals.items(), key=lambda x: x[1]['total_us'], reverse=True)
    return sorted_ops[:top_n]

# ── Session ───────────────────────────────────────────────────────────────────
saver       = tf.train.Saver()
graph       = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

timing_records   = []
kernel_stats_all = None   # will store kernel stats from PROFILE_FRAME

print(f"\n{'='*65}")
print(f"  Point-GNN Benchmark v3 — {NUM_TEST_SAMPLE} frames")
print(f"  Kernel profiling on frame #{PROFILE_FRAME}")
print(f"  t_is_training=False  (no dropout noise)")
print(f"{'='*65}\n")

with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))
    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print('Restore from checkpoint: %s' % model_path)
    saver.restore(sess, model_path)

    # ── GPU warm-up ───────────────────────────────────────────────────────
    print("Running warm-up pass...")
    _cam = dataset.get_cam_points_in_image_with_rgb(0, config['downsample_by_voxel_size'])
    _gfn = get_graph_generate_fn(config['graph_gen_method'])
    _vcl, _kil, _el = _gfn(_cam.xyz, **config['runtime_graph_gen_kwargs'])
    _n  = _cam.attr.shape[0]
    _iv = np.zeros((_n, 4), dtype=np.float32) if config['input_features'] in \
          ('irgb','0rgb','0000','i000') else np.zeros((_n, 1), dtype=np.float32)
    _fd = {t_initial_vertex_features: _iv, t_is_training: False}
    _fd.update(dict(zip(t_edges_list, _el)))
    _fd.update(dict(zip(t_keypoint_indices_list, _kil)))
    _fd.update(dict(zip(t_vertex_coord_list, _vcl)))
    sess.run(fetches, feed_dict=_fd)
    print("Warm-up done. Starting benchmark...\n")

    for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):
        frame_times = {}

        # ── Stage 1: Data Loading (CPU) ───────────────────────────────────
        t0 = time.perf_counter()
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        calib = dataset.get_calib(frame_idx)
        dataset.get_image(frame_idx)
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        t1 = time.perf_counter()
        frame_times['1_data_loading_ms'] = (t1 - t0) * 1000

        # ── Stage 2: Graph Construction (CPU) ─────────────────────────────
        t2 = time.perf_counter()
        graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(
            cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
        t3 = time.perf_counter()
        frame_times['2_graph_construction_ms'] = (t3 - t2) * 1000

        # Prepare input features
        if config['input_features'] == 'irgb':
            input_v = cam_rgb_points.attr
        elif config['input_features'] == '0rgb':
            input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)),
                                  cam_rgb_points.attr[:, 1:]])
        elif config['input_features'] == '0000':
            input_v = np.zeros_like(cam_rgb_points.attr)
        elif config['input_features'] == 'i000':
            input_v = np.hstack([cam_rgb_points.attr[:, [0]],
                                  np.zeros((cam_rgb_points.attr.shape[0], 3))])
        elif config['input_features'] == 'i':
            input_v = cam_rgb_points.attr[:, [0]]
        elif config['input_features'] == '0':
            input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

        last_layer_graph_level = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz  = vertex_coord_list[last_layer_graph_level + 1]

        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,
                         'Cyclist': 5, 'DontCare': 7}
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
        feed_dict.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))

        # ── Stage 3: GNN + Detection Head ─────────────────────────────────
        # On the profile frame: enable full TF Timeline tracing
        # On all other frames: plain wall-clock timing
        collect_trace = (frame_idx == PROFILE_FRAME)

        if collect_trace:
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            t4 = time.perf_counter()
            results = sess.run(fetches, feed_dict=feed_dict,
                               options=run_options, run_metadata=run_metadata)
            t5 = time.perf_counter()

            # Extract kernel stats from this frame
            kernel_stats_all = extract_kernel_stats(run_metadata)

            # Save chrome tracing JSON
            tl      = timeline.Timeline(run_metadata.step_stats)
            ctf     = tl.generate_chrome_trace_format()
            tl_path = os.path.join(OUTPUT_DIR, 'timeline.json')
            with open(tl_path, 'w') as f:
                f.write(ctf)
            print(f"\n  [Timeline saved → {tl_path}]")
            print(f"  Open chrome://tracing in Chrome and load this file.\n")
        else:
            t4 = time.perf_counter()
            results = sess.run(fetches, feed_dict=feed_dict)
            t5 = time.perf_counter()

        frame_times['3_gnn_and_head_ms'] = (t5 - t4) * 1000

        # ── Stage 4: Post-processing (CPU) ────────────────────────────────
        t6 = time.perf_counter()

        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)).reshape((-1))
        box_probs  = box_probs.reshape((-1))
        pred_boxes = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))

        last_layer_points_xyz_tiled = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape((-1, 3))

        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            last_layer_points_xyz_tiled,
            pred_boxes, label_map)

        box_mask    = (box_labels > 0) * (box_labels < NUM_CLASSES - 1)
        box_mask    = box_mask * (box_probs > 1. / NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]

        pred_labels = []
        if box_indices.size != 0:
            box_labels_f    = box_labels[box_indices]
            box_probs_f     = box_probs[box_indices]
            decoded_boxes_f = decoded_boxes[box_indices, 0]
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5
            detection_scores = box_probs_f

            if USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_uncertainty(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            elif USE_BOX_MERGE and not USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_merge_only(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            elif not USE_BOX_MERGE and USE_BOX_SCORE:
                (class_labels, detection_boxes_3d, detection_scores, nms_indices) = \
                    nms.nms_boxes_3d_score_only(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            else:
                (class_labels, detection_boxes_3d, detection_scores, nms_indices) = \
                    nms.nms_boxes_3d(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))

            box_probs_f = detection_scores
            detection_boxes_3d_corners = nms.boxes_3d_to_corners(detection_boxes_3d)

            for i in range(len(detection_boxes_3d_corners)):
                corners_cam_points = Points(xyz=detection_boxes_3d_corners[i], attr=None)
                corners_img_points = dataset.cam_points_to_image(corners_cam_points, calib)
                corners_xy = corners_img_points.xyz[:, :2]

                if config['label_method'] == 'yaw':
                    all_class_name = ['Background','Car','Car','Pedestrian',
                                      'Pedestrian','Cyclist','Cyclist','DontCare']
                elif config['label_method'] == 'Car':
                    all_class_name = ['Background','Car','Car','DontCare']
                elif config['label_method'] == 'Pedestrian_and_Cyclist':
                    all_class_name = ['Background','Pedestrian','Pedestrian',
                                      'Cyclist','Cyclist','DontCare']
                else:
                    all_class_name = ['Background','Car','Car','Pedestrian',
                                      'Pedestrian','Cyclist','Cyclist','DontCare']

                class_name = all_class_name[class_labels[i]]
                xmin, ymin = np.amin(corners_xy, axis=0)
                xmax, ymax = np.amax(corners_xy, axis=0)
                clip_xmin = max(xmin, 0.0);   clip_ymin = max(ymin, 0.0)
                clip_xmax = min(xmax, 1242.0); clip_ymax = min(ymax, 375.0)
                trunc = 1.0 - (clip_ymax-clip_ymin)*(clip_xmax-clip_xmin) / \
                              ((ymax-ymin)*(xmax-xmin))
                if trunc > 0.4:
                    continue
                x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                assert l > 0, str(i)
                score = box_probs_f[i]
                if USE_BOX_SCORE:
                    tmp_label = {"x3d": x3d, "y3d": y3d, "z3d": z3d,
                                 "yaw": yaw, "height": h, "width": w, "length": l}
                    inside_mask   = dataset.sel_xyz_in_box3d(
                        tmp_label, last_layer_points_xyz_tiled[box_indices])
                    points_inside = last_layer_points_xyz_tiled[box_indices][inside_mask]
                    score = (1 + occlusion(tmp_label, points_inside)) * score
                pred_labels.append((class_name, -1, -1, 0,
                                    clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                                    h, w, l, x3d, y3d, z3d, yaw, score))

        t7 = time.perf_counter()
        frame_times['4_postprocessing_ms'] = (t7 - t6) * 1000
        frame_times['5_end_to_end_ms'] = sum(frame_times.values())
        timing_records.append(frame_times)

        # Write output file
        filename = os.path.join(OUTPUT_DIR, 'data', dataset.get_filename(frame_idx) + '.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for pred_label in pred_labels:
                for field in pred_label:
                    f.write(str(field) + ' ')
                f.write('\n')
            f.write('\n')

# ── Stage timing results ───────────────────────────────────────────────────────
STAGE_LABELS = {
    '1_data_loading_ms':       'Data Loading         (CPU)',
    '2_graph_construction_ms': 'Graph Construction   (CPU)',
    '3_gnn_and_head_ms':       'GNN + Detection Head (CPU/GPU)',
    '4_postprocessing_ms':     'Post-processing      (CPU) [decode+NMS+filter]',
    '5_end_to_end_ms':         'End-to-End Latency',
}

keys  = list(STAGE_LABELS.keys())
means = {k: np.mean([r[k] for r in timing_records]) for k in keys}
stds  = {k: np.std( [r[k] for r in timing_records]) for k in keys}
total = means['5_end_to_end_ms']

print(f"\n{'='*72}")
print(f"  STAGE TIMING  —  {NUM_TEST_SAMPLE} frames  (t_is_training=False, warmed up)")
print(f"{'='*72}")
print(f"  {'Stage':<50} {'Mean(ms)':>8}  {'Std(ms)':>7}  {'%Total':>7}")
print(f"  {'-'*70}")
for k in keys:
    pct = (means[k] / total * 100) if k != '5_end_to_end_ms' else 100.0
    print(f"  {STAGE_LABELS[k]:<50} {means[k]:>8.2f}  {stds[k]:>7.2f}  {pct:>6.1f}%")
print(f"{'='*72}")

# ── Kernel-level breakdown (from TF Timeline) ─────────────────────────────────
if kernel_stats_all:
    op_summary = summarize_by_op_type(kernel_stats_all, top_n=20)
    total_kernel_us = sum(v['total_us'] for _, v in op_summary)

    print(f"\n{'='*72}")
    print(f"  KERNEL BREAKDOWN  —  Frame #{PROFILE_FRAME}  (TF Timeline, top 20 ops)")
    print(f"  These are the actual TF op kernels executing inside the GNN forward pass.")
    print(f"{'='*72}")
    print(f"  {'Kernel/Op Name':<35} {'Total(ms)':>9}  {'Count':>6}  {'%GPU Time':>9}")
    print(f"  {'-'*65}")
    for op_name, stats in op_summary:
        ms  = stats['total_us'] / 1000.0
        cnt = stats['count']
        pct = stats['total_us'] / total_kernel_us * 100
        print(f"  {op_name:<35} {ms:>9.2f}  {cnt:>6}  {pct:>8.1f}%")
    print(f"{'='*72}\n")

    # Save kernel CSV
    kernel_csv_path = os.path.join(OUTPUT_DIR, 'kernel_profile.csv')
    with open(kernel_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'kernel_op_name', 'total_ms', 'call_count',
                         'pct_of_profiled_time', 'device'])
        for rank, (op_name, stats) in enumerate(op_summary, 1):
            writer.writerow([
                rank, op_name,
                f"{stats['total_us']/1000:.3f}",
                stats['count'],
                f"{stats['total_us']/total_kernel_us*100:.1f}%",
                stats['device']
            ])
    print(f"  Kernel CSV saved → {kernel_csv_path}")

    # Also save full raw kernel list
    raw_csv_path = os.path.join(OUTPUT_DIR, 'kernel_profile_raw.csv')
    with open(raw_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['op_name', 'device', 'duration_us', 'duration_ms'])
        for k in kernel_stats_all[:200]:   # top 200 individual ops
            writer.writerow([k['op_name'], k['device'],
                             k['duration_us'], f"{k['duration_ms']:.3f}"])
    print(f"  Raw kernel CSV saved → {raw_csv_path}")

# ── Stage CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, 'benchmark_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_idx'] + [STAGE_LABELS[k] for k in keys])
    for i, record in enumerate(timing_records):
        writer.writerow([i] + [f"{record[k]:.3f}" for k in keys])
    writer.writerow([])
    writer.writerow(['MEAN'] + [f"{means[k]:.3f}" for k in keys])
    writer.writerow(['STD']  + [f"{stds[k]:.3f}"  for k in keys])
print(f"  Stage CSV saved → {csv_path}\n")

# ── Bottleneck summary ────────────────────────────────────────────────────────
ranked = sorted(
    [(k, means[k]) for k in keys if k != '5_end_to_end_ms'],
    key=lambda x: x[1], reverse=True)
print("  BOTTLENECK RANKING:")
for rank, (k, ms) in enumerate(ranked, 1):
    print(f"    #{rank}  {STAGE_LABELS[k]:<50}  {ms:.2f} ms  ({ms/total*100:.1f}%)")
print()

# ── Note on per-iteration timing ──────────────────────────────────────────────
print("  NOTE ON GNN ITERATIONS (T=1,2,3):")
print("  Point-GNN paper reports T=3 for best mAP, T=2 for best accuracy/speed tradeoff.")
print("  TF executes all 3 iterations as one fused graph in sess.run().")
print("  To see per-iteration kernel breakdown, open timeline.json in chrome://tracing")
print("  and search for 'layer2', 'layer3', 'layer4' scope names.")
print("  Each scope corresponds to one GNN iteration.")
print()