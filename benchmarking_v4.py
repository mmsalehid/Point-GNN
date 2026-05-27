"""
Point-GNN Benchmarking v3 — kernel-level profiling
====================================================
5-Stage Benchmark + TF Timeline kernel breakdown:
  1. Data Loading
  2. Graph Construction
  3. GNN + Detection Head
  4. Post-processing
  5. End-to-End Latency
"""

import os
import time
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from util.tf_compat import tf
from tensorflow.python.client import timeline
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_encoding_len
from models import nms
from util.config_util import load_config


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Point-GNN kernel benchmarking')
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--benchmark_frames', type=int, default=10)
parser.add_argument('--profile_frame', type=int, default=5,
                    help='Which frame to collect TF Timeline kernel trace on')
args = parser.parse_args()

IS_TEST          = args.test
USE_BOX_MERGE    = args.use_box_merge
USE_BOX_SCORE    = args.use_box_score
DATASET_DIR      = args.dataset_root_dir
BENCHMARK_FRAMES = args.benchmark_frames
PROFILE_FRAME    = args.profile_frame

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

NUM_TEST_SAMPLE = min(BENCHMARK_FRAMES, dataset.num_files)
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
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
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


# ── Kernel extraction helpers ─────────────────────────────────────────────────
def extract_kernel_stats(run_metadata):
    kernel_stats = []
    for dev_stats in run_metadata.step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
            duration = node_stats.op_end_rel_micros - node_stats.op_start_rel_micros
            if duration > 0:
                kernel_stats.append({
                    'op_name':     node_stats.node_name,
                    'device':      dev_stats.device,
                    'duration_us': duration,
                    'duration_ms': duration / 1000.0,
                })
    return sorted(kernel_stats, key=lambda x: x['duration_us'], reverse=True)


def summarize_by_op_type(kernel_stats, top_n=20):
    op_totals = {}
    for k in kernel_stats:
        base = k['op_name'].split('/')[-1].split(':')[0]
        if base not in op_totals:
            op_totals[base] = {'total_us': 0, 'count': 0, 'device': k['device']}
        op_totals[base]['total_us'] += k['duration_us']
        op_totals[base]['count']    += 1
    return sorted(op_totals.items(), key=lambda x: x[1]['total_us'], reverse=True)[:top_n]


# ── Session ───────────────────────────────────────────────────────────────────
saver       = tf.train.Saver()
graph       = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

timing_records   = []
kernel_stats_all = None
raw_csv_path     = None

print(f"\n{'='*65}")
print(f"  Point-GNN Benchmark v3 — {NUM_TEST_SAMPLE} frames")
print(f"  Kernel profiling on frame #{PROFILE_FRAME}")
print(f"  NO visualization (pure timing only)")
print(f"{'='*65}\n")

with tf.Session(graph=graph,
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restoring from checkpoint: {model_path}')
    saver.restore(sess, model_path)

    # ── GPU warm-up ───────────────────────────────────────────────────────────
    print("Running warm-up pass...")
    _cam = dataset.get_cam_points_in_image_with_rgb(
        0, config['downsample_by_voxel_size'])
    _gfn = get_graph_generate_fn(config['graph_gen_method'])
    _vcl, _kil, _el = _gfn(_cam.xyz, **config['runtime_graph_gen_kwargs'])
    _n  = _cam.attr.shape[0]
    _iv = (np.zeros((_n, 4), dtype=np.float32)
           if config['input_features'] in ('irgb','0rgb','0000','i000')
           else np.zeros((_n, 1), dtype=np.float32))
    _fd = {t_initial_vertex_features: _iv, t_is_training: False}
    _fd.update(dict(zip(t_edges_list, _el)))
    _fd.update(dict(zip(t_keypoint_indices_list, _kil)))
    _fd.update(dict(zip(t_vertex_coord_list, _vcl)))
    sess.run(fetches, feed_dict=_fd)
    print("Warm-up done. Starting benchmark...\n")

    # ── Main benchmark loop ───────────────────────────────────────────────────
    for frame_idx in tqdm(range(NUM_TEST_SAMPLE)):
        frame_times = {}

        # Stage 1 — Data Loading
        t0 = time.perf_counter()
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        dataset.get_calib(frame_idx)
        dataset.get_image(frame_idx)
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        t1 = time.perf_counter()
        frame_times['1_data_loading_ms'] = (t1 - t0) * 1000

        # Stage 2 — Graph Construction
        t2 = time.perf_counter()
        graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list,
         keypoint_indices_list,
         edges_list) = graph_generate_fn(
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
        else:
            input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

        last_layer_graph_level = \
            config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]

        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1,
                         'Pedestrian': 3, 'Cyclist': 5, 'DontCare': 7}
        elif config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        else:
            label_map = {'Background': 0, 'Pedestrian': 1,
                         'Cyclist': 3, 'DontCare': 5}

        feed_dict = {t_initial_vertex_features: input_v, t_is_training: False}
        feed_dict.update(dict(zip(t_edges_list, edges_list)))
        feed_dict.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))

        # Stage 3 — GNN + Detection Head
        collect_trace = (frame_idx == PROFILE_FRAME)

        if collect_trace:
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            t4 = time.perf_counter()
            results = sess.run(fetches, feed_dict=feed_dict,
                               options=run_options,
                               run_metadata=run_metadata)
            t5 = time.perf_counter()
            kernel_stats_all = extract_kernel_stats(run_metadata)
            print(f"\n  [Kernel trace collected — frame #{frame_idx}]")
        else:
            t4 = time.perf_counter()
            results = sess.run(fetches, feed_dict=feed_dict)
            t5 = time.perf_counter()

        frame_times['3_gnn_and_head_ms'] = (t5 - t4) * 1000

        # Stage 4 — Post-processing
        t6 = time.perf_counter()

        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)).reshape((-1))
        box_probs  = box_probs.reshape((-1))
        pred_boxes = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))

        last_layer_xyz_tiled = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape((-1, 3))

        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            last_layer_xyz_tiled, pred_boxes, label_map)

        box_mask    = (box_labels > 0) * (box_labels < NUM_CLASSES - 1)
        box_mask    = box_mask * (box_probs > 1. / NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]

        if box_indices.size != 0:
            box_labels_f    = box_labels[box_indices].copy()
            box_probs_f     = box_probs[box_indices].copy()
            decoded_boxes_f = decoded_boxes[box_indices, 0]
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5
            detection_scores = box_probs_f

            if USE_BOX_MERGE and USE_BOX_SCORE:
                (_, detection_boxes_3d, detection_scores, _) = \
                    nms.nms_boxes_3d_uncertainty(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            else:
                (_, detection_boxes_3d, detection_scores, _) = \
                    nms.nms_boxes_3d(
                        box_labels_f, decoded_boxes_f, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))

        t7 = time.perf_counter()
        frame_times['4_postprocessing_ms'] = (t7 - t6) * 1000
        frame_times['5_end_to_end_ms']     = (t7 - t0) * 1000
        timing_records.append(frame_times)

        # Write output file
        filename = os.path.join(
            OUTPUT_DIR, 'data',
            dataset.get_filename(frame_idx) + '.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write('\n')


# ── Stage timing results ──────────────────────────────────────────────────────
STAGE_LABELS = {
    '1_data_loading_ms':       'Data Loading         (CPU)',
    '2_graph_construction_ms': 'Graph Construction   (CPU)',
    '3_gnn_and_head_ms':       'GNN + Detection Head (CPU/GPU)',
    '4_postprocessing_ms':     'Post-processing      (CPU)',
    '5_end_to_end_ms':         'End-to-End Latency',
}

keys  = list(STAGE_LABELS.keys())
means = {k: np.mean([r[k] for r in timing_records]) for k in keys}
stds  = {k: np.std( [r[k] for r in timing_records]) for k in keys}
total = means['5_end_to_end_ms']

print(f"\n{'='*72}")
print(f"  STAGE TIMING — {NUM_TEST_SAMPLE} frames (t_is_training=False, warmed up)")
print(f"{'='*72}")
print(f"  {'Stage':<50} {'Mean(ms)':>8}  {'Std(ms)':>7}  {'%Total':>7}")
print(f"  {'-'*70}")
for k in keys:
    pct = (means[k] / total * 100) if k != '5_end_to_end_ms' else 100.0
    print(f"  {STAGE_LABELS[k]:<50} {means[k]:>8.2f}  "
          f"{stds[k]:>7.2f}  {pct:>6.1f}%")
print(f"{'='*72}")


# ── Kernel breakdown ──────────────────────────────────────────────────────────
if kernel_stats_all:
    op_summary       = summarize_by_op_type(kernel_stats_all, top_n=20)
    total_kernel_us  = sum(v['total_us'] for _, v in op_summary)

    print(f"\n{'='*72}")
    print(f"  KERNEL BREAKDOWN — Frame #{PROFILE_FRAME} (TF Timeline, top 20 ops)")
    print(f"{'='*72}")
    print(f"  {'Kernel/Op Name':<35} {'Total(ms)':>9}  {'Count':>6}  {'% Time':>8}")
    print(f"  {'-'*65}")
    for op_name, stats in op_summary:
        ms  = stats['total_us'] / 1000.0
        cnt = stats['count']
        pct = stats['total_us'] / total_kernel_us * 100
        print(f"  {op_name:<35} {ms:>9.2f}  {cnt:>6}  {pct:>7.1f}%")
    print(f"{'='*72}\n")

    # Save CSVs
    kernel_csv_path = os.path.join(OUTPUT_DIR, 'kernel_profile.csv')
    raw_csv_path    = os.path.join(OUTPUT_DIR, 'kernel_profile_raw.csv')

    with open(kernel_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank','kernel_op_name','total_ms',
                         'call_count','pct_of_profiled_time'])
        for rank, (op_name, stats) in enumerate(op_summary, 1):
            writer.writerow([
                rank, op_name,
                f"{stats['total_us']/1000:.3f}",
                stats['count'],
                f"{stats['total_us']/total_kernel_us*100:.1f}%",
            ])

    with open(raw_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['op_name','duration_us','duration_ms'])
        for k in kernel_stats_all[:200]:
            writer.writerow([k['op_name'], k['duration_us'],
                             f"{k['duration_ms']:.3f}"])

    print(f"  Kernel CSV    → {kernel_csv_path}")
    print(f"  Raw CSV       → {raw_csv_path}")

    # ── MATHEMATICAL OPERATION CLASSIFICATION PLOT ────────────────────────────
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['op_type'] = df_raw['op_name'].apply(
        lambda x: x.split('/')[-1])

    # SpMM: sparse matrix operations — irregular memory, graph-topology dependent
    spmm_ops = {
        'GatherV2',
        'GatherV2_1',
        'GatherV2_2',
        'GatherV2_3',
        'scatter_max',
        'strided_slice',
        'strided_slice_1',
        'strided_slice_2',
    }

    # GeMM: dense matrix multiply — regular memory, fully parallelizable
    gemm_ops = {
        'MatMul',
        'BiasAdd',
        'Relu',
        'weights',
        'biases',
        'read',
    }

    # Other: data movement and bookkeeping
    other_ops = {
        'sub',
        'add',
        'add_1',
        'concat',
        'Softmax',
        'ArgMax',
        'Shape',
        'stack',
    }

    def get_group_time(df, op_set):
        mask = df['op_type'].isin(op_set)
        return df[mask]['duration_ms'].sum()

    def get_op_breakdown(df, op_set):
        mask = df['op_type'].isin(op_set)
        return df[mask].groupby('op_type')['duration_ms'].sum().sort_values(
            ascending=False)

    t_spmm  = get_group_time(df_raw, spmm_ops)
    t_gemm  = get_group_time(df_raw, gemm_ops)
    t_other = get_group_time(df_raw, other_ops)
    t_total = df_raw['duration_ms'].sum()

    all_classified = spmm_ops | gemm_ops | other_ops
    unclassified = df_raw[~df_raw['op_type'].isin(all_classified)]
    if len(unclassified) > 0:
        t_unclassified = unclassified['duration_ms'].sum()
        print(f"\n  Unclassified kernels ({t_unclassified:.1f}ms):")
        for op, t in unclassified.groupby('op_type')['duration_ms'].sum(
                ).sort_values(ascending=False).items():
            print(f"    {op:<30} {t:.1f}ms")
    else:
        t_unclassified = 0.0

    print(f"\n{'='*65}")
    print(f"  MATHEMATICAL OPERATION CLASSIFICATION")
    print(f"{'='*65}")
    print(f"  Total profiled time: {t_total:.1f}ms")
    print(f"\n  {'Group':<10} {'Time(ms)':>10} {'% of total':>12}")
    print(f"  {'-'*36}")
    print(f"  {'SpMM':<10} {t_spmm:>10.1f} {t_spmm/t_total*100:>11.1f}%")
    print(f"  {'GeMM':<10} {t_gemm:>10.1f} {t_gemm/t_total*100:>11.1f}%")
    print(f"  {'Other':<10} {t_other:>10.1f} {t_other/t_total*100:>11.1f}%")
    if t_unclassified > 0:
        print(f"  {'Unclassified':<10} {t_unclassified:>10.1f} "
              f"{t_unclassified/t_total*100:>11.1f}%")

    print(f"\n  SpMM breakdown (all sparse ops):")
    for op, t in get_op_breakdown(df_raw, spmm_ops).items():
        print(f"    {op:<25} {t:>8.1f}ms  "
              f"({t/t_spmm*100:.1f}% of SpMM)")

    print(f"\n  GeMM breakdown (all dense ops):")
    for op, t in get_op_breakdown(df_raw, gemm_ops).items():
        print(f"    {op:<25} {t:>8.1f}ms  "
              f"({t/t_gemm*100:.1f}% of GeMM)")

    print(f"\n  Other breakdown (data movement):")
    for op, t in get_op_breakdown(df_raw, other_ops).items():
        print(f"    {op:<25} {t:>8.1f}ms  "
              f"({t/t_other*100:.1f}% of Other)")
    print(f"{'='*65}\n")

    # ── Figure 1: Three-bar SpMM / GeMM / Other ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(
        'Point-GNN — Mathematical Operation Classification\n'
        'SpMM (Sparse Matrix) | GeMM (Dense Matrix) | Other (Data Movement)',
        fontsize=14, fontweight='bold')

    groups = {
        'SpMM\n(GatherV2 + scatter_max\n+ StridedSlice)': {
            'time':  t_spmm,
            'color': '#FF8C00',
            'breakdown': get_op_breakdown(df_raw, spmm_ops),
            'note': 'Irregular memory access\nThread divergence on GPU\nGraph-topology dependent',
        },
        'GeMM\n(MatMul + BiasAdd\n+ ReLU)': {
            'time':  t_gemm,
            'color': '#7B1FA2',
            'breakdown': get_op_breakdown(df_raw, gemm_ops),
            'note': 'Dense matrix multiply\nFully parallelizable\nGPU-friendly (cuBLAS)',
        },
        'Other\n(sub + concat\n+ add + misc)': {
            'time':  t_other,
            'color': '#607D8B',
            'breakdown': get_op_breakdown(df_raw, other_ops),
            'note': 'Data movement only\nNo compute bottleneck\nNeither SpMM nor GeMM',
        },
    }

    ax_main = axes[0]
    names  = list(groups.keys())
    times  = [groups[n]['time'] for n in names]
    colors = [groups[n]['color'] for n in names]

    bars = ax_main.bar(range(len(names)), times,
                       color=colors, edgecolor='black',
                       linewidth=0.8, width=0.5)

    for bar, t in zip(bars, times):
        pct = t / t_total * 100
        ax_main.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(times)*0.02,
            f'{t:.1f}ms\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold')

    ax_main.set_xticks(range(len(names)))
    ax_main.set_xticklabels(names, fontsize=9)
    ax_main.set_ylabel('Total Time (ms)', fontsize=11)
    ax_main.set_title('All Kernels Classified\nby Mathematical Operation',
                      fontsize=11, fontweight='bold')
    ax_main.set_ylim(0, max(times) * 1.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    for i, (name, info) in enumerate(groups.items()):
        ax_main.text(
            i, -max(times)*0.18,
            info['note'],
            ha='center', va='top',
            fontsize=7, color='#444444',
            style='italic')

    ax_spmm = axes[1]
    spmm_breakdown = get_op_breakdown(df_raw, spmm_ops)
    spmm_ops_present = list(spmm_breakdown.index)
    spmm_times_list  = list(spmm_breakdown.values)

    spmm_colors = {
        'GatherV2':        '#FF8C00',
        'GatherV2_1':      '#FFA333',
        'GatherV2_2':      '#FFB966',
        'GatherV2_3':      '#FFCF99',
        'scatter_max':     '#E65100',
        'strided_slice':   '#CC7000',
        'strided_slice_1': '#995400',
        'strided_slice_2': '#663800',
    }

    bars2 = ax_spmm.bar(
        range(len(spmm_ops_present)),
        spmm_times_list,
        color=[spmm_colors.get(op, '#FF8C00') for op in spmm_ops_present],
        edgecolor='black', linewidth=0.7)

    for bar, t in zip(bars2, spmm_times_list):
        if t > 1.0:
            pct = t / t_spmm * 100
            ax_spmm.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + t_spmm*0.01,
                f'{t:.1f}ms\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8)

    ax_spmm.set_xticks(range(len(spmm_ops_present)))
    ax_spmm.set_xticklabels(spmm_ops_present,
                             rotation=30, ha='right', fontsize=8)
    ax_spmm.set_ylabel('Time (ms)', fontsize=10)
    ax_spmm.set_title(
        f'SpMM Breakdown\nTotal: {t_spmm:.1f}ms '
        f'({t_spmm/t_total*100:.1f}% of all kernels)',
        fontsize=10, fontweight='bold', color='#FF8C00')
    ax_spmm.set_ylim(0, max(spmm_times_list)*1.35)
    ax_spmm.spines['top'].set_visible(False)
    ax_spmm.spines['right'].set_visible(False)

    ax_gemm = axes[2]
    gemm_breakdown = get_op_breakdown(df_raw, gemm_ops)
    gemm_ops_present = list(gemm_breakdown.index)
    gemm_times_list  = list(gemm_breakdown.values)

    gemm_colors_map = {
        'Relu':    '#7B1FA2',
        'MatMul':  '#9C27B0',
        'BiasAdd': '#CE93D8',
        'weights': '#E1BEE7',
        'biases':  '#F3E5F5',
        'read':    '#EDE7F6',
    }

    bars3 = ax_gemm.bar(
        range(len(gemm_ops_present)),
        gemm_times_list,
        color=[gemm_colors_map.get(op, '#9C27B0') for op in gemm_ops_present],
        edgecolor='black', linewidth=0.7)

    for bar, t in zip(bars3, gemm_times_list):
        if t > 0.5:
            pct = t / t_gemm * 100
            ax_gemm.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + t_gemm*0.01,
                f'{t:.1f}ms\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8)

    ax_gemm.set_xticks(range(len(gemm_ops_present)))
    ax_gemm.set_xticklabels(gemm_ops_present,
                             rotation=30, ha='right', fontsize=8)
    ax_gemm.set_ylabel('Time (ms)', fontsize=10)
    ax_gemm.set_title(
        f'GeMM Breakdown\nTotal: {t_gemm:.1f}ms '
        f'({t_gemm/t_total*100:.1f}% of all kernels)',
        fontsize=10, fontweight='bold', color='#7B1FA2')
    ax_gemm.set_ylim(0, max(gemm_times_list)*1.35)
    ax_gemm.spines['top'].set_visible(False)
    ax_gemm.spines['right'].set_visible(False)

    plt.tight_layout()
    math_plot_path = os.path.join(OUTPUT_DIR, 'math_op_classification.png')
    plt.savefig(math_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Math op classification plot → {math_plot_path}")

    # ── Figure 2: CPU vs GPU bottleneck shift ─────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.suptitle(
        'CPU vs GPU Bottleneck Shift\n'
        'Why SpMM Dominates on GPU Despite Being Small on CPU',
        fontsize=13, fontweight='bold')

    categories = ['SpMM\n(GatherV2+scatter_max)',
                  'GeMM\n(MatMul+BiasAdd+ReLU)',
                  'Other\n(concat+sub+add)']

    cpu_times = [t_spmm, t_gemm, t_other]

    gpu_factor_gemm  = 30.0
    gpu_factor_spmm  = 2.5
    gpu_factor_other = 7.0

    gpu_times = [
        t_spmm  / gpu_factor_spmm,
        t_gemm  / gpu_factor_gemm,
        t_other / gpu_factor_other,
    ]

    x      = np.arange(len(categories))
    width  = 0.35
    colors_cpu = ['#FF8C00', '#7B1FA2', '#607D8B']
    colors_gpu = ['#FFD580', '#CE93D8', '#B0BEC5']

    bars_cpu = ax.bar(x - width/2, cpu_times, width,
                      label='CPU (measured)',
                      color=colors_cpu,
                      edgecolor='black', linewidth=0.7)
    bars_gpu = ax.bar(x + width/2, gpu_times, width,
                      label='GPU (estimated)',
                      color=colors_gpu,
                      edgecolor='black', linewidth=0.7,
                      hatch='///')

    for bar, t in zip(bars_cpu, cpu_times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f'{t:.0f}ms',
                ha='center', va='bottom', fontsize=9)

    for bar, t in zip(bars_gpu, gpu_times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f'~{t:.0f}ms',
                ha='center', va='bottom', fontsize=9,
                color='#444444')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate(
        'GeMM collapses on GPU\n→ SpMM becomes dominant bottleneck',
        xy=(0 + width/2, gpu_times[0]),
        xytext=(1.2, max(cpu_times)*0.7),
        fontsize=9, color='#FF8C00',
        arrowprops=dict(arrowstyle='->', color='#FF8C00'),
        fontweight='bold')

    ax.annotate(
        'GeMM: ~30x speedup\nfrom cuBLAS',
        xy=(1 + width/2, gpu_times[1]),
        xytext=(1.8, max(cpu_times)*0.4),
        fontsize=9, color='#7B1FA2',
        arrowprops=dict(arrowstyle='->', color='#7B1FA2'),
        fontweight='bold')

    plt.tight_layout()
    gpu_plot_path = os.path.join(OUTPUT_DIR, 'cpu_vs_gpu_bottleneck.png')
    plt.savefig(gpu_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  CPU vs GPU bottleneck plot → {gpu_plot_path}")

    # ── Graph kernel benchmark ────────────────────────────────────────────────
    df = pd.read_csv(raw_csv_path)
    df['op_type'] = df['op_name'].apply(lambda x: x.split('/')[-1])

    op_to_group = {
        'GatherV2':          'GATHER',
        'GatherV2_1':        'GATHER',
        'GatherV2_2':        'GATHER',
        'GatherV2_3':        'GATHER',
        'strided_slice':     'GATHER',
        'strided_slice_1':   'GATHER',
        'strided_slice_2':   'GATHER',
        'concat':            'CONCAT',
        'sub':               'CONCAT',
        'add':               'CONCAT',
        'add_1':             'CONCAT',
        'scatter_max':       'SCATTER',
        'scatter_mean':      'SCATTER',
        'scatter_add':       'SCATTER',
        'MatMul':            'LINEAR',
        'BatchMatMul':       'LINEAR',
        'FusedMatMul':       'LINEAR',
        '_MklMatMul':        'LINEAR',
        '_MklFusedMatMul':   'LINEAR',
        'BiasAdd':           'LINEAR',
    }

    individual_times = {}
    for op_type, group in op_to_group.items():
        t = df[df['op_type'] == op_type]['duration_ms'].sum()
        if t > 0:
            individual_times[op_type] = {'time': t, 'group': group}

    group_times = {'GATHER': 0, 'CONCAT': 0, 'SCATTER': 0, 'LINEAR': 0}
    for op_type, info in individual_times.items():
        group_times[info['group']] += info['time']

    total_profiled = df['duration_ms'].sum()

    group_op_order = {
        'GATHER':  ['GatherV2', 'GatherV2_1', 'GatherV2_2', 'GatherV2_3',
                    'strided_slice', 'strided_slice_1', 'strided_slice_2'],
        'CONCAT':  ['sub', 'add', 'add_1', 'concat'],
        'SCATTER': ['scatter_max', 'scatter_mean', 'scatter_add'],
        'LINEAR':  ['MatMul', 'BiasAdd'],
    }

    group_headers = {
        'GATHER':  'GATHER  (GatherV2 + StridedSlice)',
        'CONCAT':  'CONCAT  (Sub + AddV2 + ConcatV2)',
        'SCATTER': 'SCATTER (Max | Mean | Add)',
        'LINEAR':  'LINEAR LAYER (MatMul + BiasAdd)',
    }

    op_colors_map = {
        'GatherV2':        '#FF8C00',
        'GatherV2_1':      '#FFA333',
        'GatherV2_2':      '#FFB966',
        'GatherV2_3':      '#FFCF99',
        'strided_slice':   '#CC7000',
        'strided_slice_1': '#995400',
        'strided_slice_2': '#663800',
        'sub':    '#2196F3',
        'add':    '#42A5F5',
        'add_1':  '#64B5F6',
        'concat': '#1565C0',
        'scatter_max':  '#2E7D32',
        'scatter_mean': '#66BB6A',
        'scatter_add':  '#A5D6A7',
        'MatMul':  '#7B1FA2',
        'BiasAdd': '#CE93D8',
    }

    # ── Chart 1: GATHER + CONCAT + SCATTER ───────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7))
    fig1.suptitle('Point-GNN Graph Kernel Benchmark\n(CPU, TF Timeline Profiling)',
                  fontsize=14, fontweight='bold')

    for group, ax in zip(['GATHER', 'CONCAT', 'SCATTER'], axes1):
        ops    = [op for op in group_op_order[group] if op in individual_times]
        times  = [individual_times[op]['time'] for op in ops]
        colors = [op_colors_map[op] for op in ops]

        if group == 'SCATTER':
            ops_display = ['scatter_max', 'scatter_mean', 'scatter_add']
            times_display = []
            for op in ops_display:
                if op in individual_times:
                    times_display.append(individual_times[op]['time'])
                else:
                    times_display.append(0)
            ops    = ops_display
            times  = times_display
            colors = [op_colors_map[op] for op in ops]

        bars = ax.bar(range(len(ops)), times,
                      color=colors, edgecolor='black', linewidth=0.7)

        for bar, t in zip(bars, times):
            g_time = group_times[group] if group_times[group] > 0 else 1
            pct    = t / g_time * 100 if t > 0 else 0
            label  = f'{t:.1f}ms\n({pct:.1f}%)' if t > 0 else 'N/A\n(not used)'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(times)*0.02 if max(times) > 0 else 0.5,
                    label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels(ops, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(
            f'{group_headers[group]}\n'
            f'Total: {group_times[group]:.1f}ms '
            f'({group_times[group]/total_profiled*100:.1f}% of total)',
            fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(times) * 1.45 if max(times) > 0 else 5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, 'graph_kernel_benchmark.png')
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Graph kernel plot → {plot1_path}")

    # ── Chart 2: LINEAR LAYER ─────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    fig2.suptitle('Point-GNN Linear Layer Benchmark\n(CPU, TF Timeline Profiling)',
                  fontsize=14, fontweight='bold')

    ops    = [op for op in group_op_order['LINEAR'] if op in individual_times]
    times  = [individual_times[op]['time'] for op in ops]
    colors = [op_colors_map[op] for op in ops]

    bars = ax2.bar(range(len(ops)), times,
                   color=colors, edgecolor='black',
                   linewidth=0.7, width=0.4)

    for bar, t in zip(bars, times):
        g_time = group_times['LINEAR'] if group_times['LINEAR'] > 0 else 1
        pct    = t / g_time * 100
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(times)*0.02,
                 f'{t:.1f}ms\n({pct:.1f}% of layer)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(range(len(ops)))
    ax2.set_xticklabels(ops, fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=11)
    ax2.set_title(
        f'{group_headers["LINEAR"]}\n'
        f'Total: {group_times["LINEAR"]:.1f}ms '
        f'({group_times["LINEAR"]/total_profiled*100:.1f}% of total)',
        fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(times) * 1.4 if times else 5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, 'linear_layer_benchmark.png')
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Linear layer plot → {plot2_path}")
