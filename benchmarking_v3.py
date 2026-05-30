"""
Point-GNN Benchmarking v3 — Full System Profiling
==================================================
- TF Timeline on EVERY frame
- Four-category taxonomy: Sparse / Dense / Memory / Other
- Per-frame normalization then mean
- CPU wall-clock timing
- Full pipeline: graph construction + inference + post-processing
- Mean/average results only — no per-frame plots
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.tf_compat import tf
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict

from dataset.kitti_dataset import KittiDataset
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_encoding_len
from models import nms
from util.config_util import load_config
from models import graph_gen


# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path',      type=str)
parser.add_argument('--test',               dest='test',
                    action='store_true',    default=False)
parser.add_argument('--no-box-merge',       dest='use_box_merge',
                    action='store_false',   default=True)
parser.add_argument('--no-box-score',       dest='use_box_score',
                    action='store_false',   default=True)
parser.add_argument('--dataset_root_dir',   type=str,
                    default='../dataset/kitti/')
parser.add_argument('--dataset_split_file', type=str, default='')
parser.add_argument('--output_dir',         type=str, default='')
parser.add_argument('--benchmark_frames',   type=int, default=10)
args = parser.parse_args()

IS_TEST          = args.test
USE_BOX_MERGE    = args.use_box_merge
USE_BOX_SCORE    = args.use_box_score
DATASET_DIR      = args.dataset_root_dir
BENCHMARK_FRAMES = args.benchmark_frames
FEAT_DIM         = 300
N_RUNS           = 5

DATASET_SPLIT_FILE = (
    args.dataset_split_file if args.dataset_split_file
    else os.path.join(DATASET_DIR, './3DOP_splits/val.txt'))

OUTPUT_DIR = (
    args.output_dir if args.output_dir
    else os.path.join(args.checkpoint_path, './eval/'))

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH     = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
config = load_config(CONFIG_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)


# ── Dataset ────────────────────────────────────────────────────────────────────
if IS_TEST:
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '', num_classes=config['num_classes'], is_training=False)
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

graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])


# ── Taxonomy ───────────────────────────────────────────────────────────────────
TAXONOMY = {
    'sparse': {
        'GatherV2', 'GatherV2_1', 'GatherV2_2', 'GatherV2_3',
        'scatter_max',
        'strided_slice', 'strided_slice_1', 'strided_slice_2',
    },
    'dense': {
        'MatMul', 'BiasAdd', 'Relu',
        'weights', 'biases', 'read',
    },
    'memory': {
        'concat', 'sub', 'add', 'add_1',
        'stack', 'Shape', 'ExpandDims',
    },
    'other': {
        'Softmax', 'ArgMax',
    },
}

TAXONOMY_COLORS = {
    'sparse': '#FF8C00',
    'dense':  '#7B1FA2',
    'memory': '#2196F3',
    'other':  '#607D8B',
}
TAXONOMY_LABELS = {
    'sparse': 'Sparse Ops\n(Gather+Scatter)',
    'dense':  'Dense Compute\n(MatMul+ReLU)',
    'memory': 'Memory Ops\n(concat+reshape)',
    'other':  'Other',
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_frame_kernels(kernel_stats):
    cat_times = defaultdict(float)
    for k in kernel_stats:
        op_base = k['op_name'].split('/')[-1].split(':')[0]
        ms      = k['duration_ms']
        placed  = False
        for cat, op_set in TAXONOMY.items():
            if op_base in op_set:
                cat_times[cat] += ms
                placed = True
                break
        if not placed:
            cat_times['other'] += ms
    classified_total = sum(cat_times.values())
    cat_pcts = {
        cat: (cat_times.get(cat, 0.0) / classified_total * 100
              if classified_total > 0 else 0.0)
        for cat in TAXONOMY
    }
    return {'times': dict(cat_times), 'pcts': cat_pcts,
            'total': classified_total}


def extract_kernel_stats(run_metadata):
    stats = []
    for dev in run_metadata.step_stats.dev_stats:
        for node in dev.node_stats:
            dur = node.op_end_rel_micros - node.op_start_rel_micros
            if dur > 0:
                stats.append({'op_name':     node.node_name,
                               'duration_ms': dur / 1000.0})
    return stats


def build_a_sel(src_idx, num_e, num_v):
    rows = np.arange(num_e, dtype=np.int32)
    data = np.ones(num_e, dtype=np.float32)
    return coo_matrix((data, (rows, src_idx)), shape=(num_e, num_v))


def spmm_coo(coo_mat, X):
    Y = np.zeros((coo_mat.shape[0], X.shape[1]), dtype=np.float32)
    np.add.at(Y, coo_mat.row, coo_mat.data[:, None] * X[coo_mat.col])
    return Y


def spmm_csr(csr_mat, X):
    return csr_mat.dot(X)


def get_input_features(cam_rgb_points, input_features_cfg):
    attr = cam_rgb_points.attr
    if input_features_cfg == 'irgb':   return attr
    elif input_features_cfg == '0rgb': return np.hstack([np.zeros((attr.shape[0], 1)), attr[:, 1:]])
    elif input_features_cfg == '0000': return np.zeros_like(attr)
    elif input_features_cfg == 'i000': return np.hstack([attr[:, [0]], np.zeros((attr.shape[0], 3))])
    elif input_features_cfg == 'i':    return attr[:, [0]]
    else:                              return np.zeros((attr.shape[0], 1))


def build_feed_dict(input_v, vertex_coord_list,
                    keypoint_indices_list, edges_list):
    fd = {t_initial_vertex_features: input_v, t_is_training: False}
    fd.update(dict(zip(t_edges_list,            edges_list)))
    fd.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
    fd.update(dict(zip(t_vertex_coord_list,     vertex_coord_list)))
    return fd


# ── Model placeholders ─────────────────────────────────────────────────────────
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_decoding_fn  = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] in ('irgb', '0000', 'i000', '0rgb'):
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
else:
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])

num_levels = len(config['runtime_graph_gen_kwargs']['level_configs'])

t_vertex_coord_list     = [tf.placeholder(dtype=tf.float32, shape=[None, 3])
                            for _ in range(num_levels + 1)]
t_edges_list            = [tf.placeholder(dtype=tf.int32,   shape=[None, 2])
                            for _ in range(num_levels)]
t_keypoint_indices_list = [tf.placeholder(dtype=tf.int32,   shape=[None, 1])
                            for _ in range(num_levels)]
t_is_training           = tf.placeholder(dtype=tf.bool, shape=[])

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

# ── Storage ────────────────────────────────────────────────────────────────────
frame_records = []
spmm_records  = []


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════════════════
saver       = tf.train.Saver()
tf_graph    = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(
        graph=tf_graph,
        config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))

    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restoring from checkpoint: {model_path}')
    saver.restore(sess, model_path)

    # Warm-up
    print("Running warm-up pass...")
    cam_w = dataset.get_cam_points_in_image_with_rgb(
        0, config['downsample_by_voxel_size'])
    vcl_w, kil_w, el_w = graph_generate_fn(
        cam_w.xyz, **config['runtime_graph_gen_kwargs'])
    iv_w = get_input_features(cam_w, config['input_features'])
    sess.run(fetches, feed_dict=build_feed_dict(iv_w, vcl_w, kil_w, el_w))
    print("Warm-up done.\n")

    for frame_idx in tqdm(range(NUM_TEST_SAMPLE), desc="Profiling frames"):

        rec = {'frame': frame_idx}

        # Stage 1: Data Loading
        t0 = time.perf_counter()
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(
            frame_idx, config['downsample_by_voxel_size'])
        dataset.get_calib(frame_idx)
        dataset.get_image(frame_idx)
        if not IS_TEST:
            dataset.get_label(frame_idx)
        t1 = time.perf_counter()
        rec['t_data_loading_ms'] = (t1 - t0) * 1000

        # Stage 2: Graph Construction
        t_gc0 = time.perf_counter()
        (vertex_coord_list,
         keypoint_indices_list,
         edges_list) = graph_generate_fn(
            cam_rgb_points.xyz,
            **config['runtime_graph_gen_kwargs'])
        t_gc1 = time.perf_counter()
        rec['t_graph_construction_ms'] = (t_gc1 - t_gc0) * 1000

        gc_downsample_ms = graph_gen._gc_timing.get('downsample_ms', 0.0)
        gc_edge_ms       = graph_gen._gc_timing.get('edge_build_ms',  0.0)
        gc_frnn_ms       = max(0.0, rec['t_graph_construction_ms']
                               - gc_downsample_ms - gc_edge_ms)

        rec['gc_sparse_ms']     = gc_frnn_ms
        rec['gc_memory_ms']     = gc_edge_ms
        rec['gc_dense_ms']      = gc_downsample_ms
        rec['gc_frnn_ms']       = gc_frnn_ms
        rec['gc_edge_build_ms'] = gc_edge_ms
        rec['gc_downsample_ms'] = gc_downsample_ms

        edges_f = edges_list[1]
        num_v_f = vertex_coord_list[1].shape[0]
        num_e_f = edges_f.shape[0]
        src_f   = edges_f[:, 0].astype(np.int32)
        rec['num_vertices'] = num_v_f
        rec['num_edges']    = num_e_f

        # Stage 2b: SpMM benchmark — CPU wall-clock only
        np.random.seed(frame_idx)
        X_f     = np.random.randn(num_v_f, FEAT_DIM).astype(np.float32)
        A_coo_f = build_a_sel(src_f, num_e_f, num_v_f)
        A_csr_f = A_coo_f.tocsr()
        t_X_f   = tf.constant(X_f)
        t_src_f = tf.constant(src_f)

        gather_t, coo_t, csr_t = [], [], []
        for _ in range(N_RUNS):
            ts = time.perf_counter()
            Y_g = sess.run(tf.gather(t_X_f, t_src_f))
            gather_t.append((time.perf_counter() - ts) * 1000)
        for _ in range(N_RUNS):
            ts = time.perf_counter()
            Y_c = spmm_coo(A_coo_f, X_f)
            coo_t.append((time.perf_counter() - ts) * 1000)
        for _ in range(N_RUNS):
            ts = time.perf_counter()
            Y_r = spmm_csr(A_csr_f, X_f)
            csr_t.append((time.perf_counter() - ts) * 1000)

        coo_match = np.allclose(Y_g, Y_c, atol=1e-4)
        csr_match = np.allclose(Y_g, Y_r, atol=1e-4)

        spmm_records.append({
            'frame':        frame_idx,
            'num_edges':    num_e_f,
            'num_vertices': num_v_f,
            'gather_mean':  np.mean(gather_t),
            'coo_mean':     np.mean(coo_t),
            'csr_mean':     np.mean(csr_t),
            'coo_match':    coo_match,
            'csr_match':    csr_match,
        })

        # Stage 3: GNN Inference — FULL_TRACE every frame
        input_v   = get_input_features(cam_rgb_points, config['input_features'])
        feed_dict = build_feed_dict(
            input_v, vertex_coord_list, keypoint_indices_list, edges_list)

        llgl = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[llgl + 1]

        run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        t4 = time.perf_counter()
        results = sess.run(
            fetches, feed_dict=feed_dict,
            options=run_options, run_metadata=run_metadata)
        t5 = time.perf_counter()
        rec['t_gnn_ms'] = (t5 - t4) * 1000

        kernel_stats         = extract_kernel_stats(run_metadata)
        frame_classification = classify_frame_kernels(kernel_stats)

        for cat in TAXONOMY:
            rec[f'gnn_{cat}_ms']  = frame_classification['times'].get(cat, 0.0)
            rec[f'gnn_pct_{cat}'] = frame_classification['pcts'][cat]
        rec['gnn_total_classified_ms'] = frame_classification['total']

        # Stage 4: Post-processing
        if config['label_method'] == 'yaw':
            label_map = {'Background':0,'Car':1,'Pedestrian':3,
                         'Cyclist':5,'DontCare':7}
        elif config['label_method'] == 'Car':
            label_map = {'Background':0,'Car':1,'DontCare':3}
        else:
            label_map = {'Background':0,'Pedestrian':1,
                         'Cyclist':3,'DontCare':5}

        t6 = time.perf_counter()
        box_probs  = results['probs']
        box_labels = np.tile(
            np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1)).reshape(-1)
        box_probs  = box_probs.reshape(-1)
        pred_boxes = results['pred_box'].reshape(-1, 1, BOX_ENCODING_LEN)
        last_xyz_tiled = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1),
            (1, NUM_CLASSES, 1)).reshape(-1, 3)

        t_bd0 = time.perf_counter()
        decoded_boxes = box_decoding_fn(
            np.expand_dims(box_labels, axis=1),
            last_xyz_tiled, pred_boxes, label_map)
        t_bd1 = time.perf_counter()
        rec['t_box_decode_ms'] = (t_bd1 - t_bd0) * 1000

        t_f0 = time.perf_counter()
        box_mask    = ((box_labels > 0)
                       * (box_labels < NUM_CLASSES - 1)
                       * (box_probs > 1. / NUM_CLASSES))
        box_indices = np.nonzero(box_mask)[0]
        t_f1 = time.perf_counter()
        rec['t_filter_ms'] = (t_f1 - t_f0) * 1000

        t_n0 = time.perf_counter()
        if box_indices.size != 0:
            box_labels_f = box_labels[box_indices].copy()
            box_probs_f  = box_probs[box_indices].copy()
            decoded_f    = decoded_boxes[box_indices, 0]
            box_labels_f[box_labels_f == 2] = 1
            box_labels_f[box_labels_f == 4] = 3
            box_labels_f[box_labels_f == 6] = 5
            if USE_BOX_MERGE and USE_BOX_SCORE:
                nms.nms_boxes_3d_uncertainty(
                    box_labels_f, decoded_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
            else:
                nms.nms_boxes_3d(
                    box_labels_f, decoded_f, box_probs_f,
                    overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                    overlapped_thres=config['nms_overlapped_thres'],
                    appr_factor=100.0, top_k=-1,
                    attributes=np.arange(len(box_indices)))
        t_n1 = time.perf_counter()
        rec['t_nms_ms'] = (t_n1 - t_n0) * 1000

        t7 = time.perf_counter()
        rec['t_post_ms']      = (t7 - t6) * 1000
        rec['post_dense_ms']  = rec['t_box_decode_ms']
        rec['post_memory_ms'] = rec['t_filter_ms']
        rec['post_other_ms']  = rec['t_nms_ms']
        rec['t_end_to_end_ms'] = (
            rec['t_data_loading_ms']
            + rec['t_graph_construction_ms']
            + rec['t_gnn_ms']
            + rec['t_post_ms'])

        frame_records.append(rec)

        filename = os.path.join(
            OUTPUT_DIR, 'data',
            dataset.get_filename(frame_idx) + '.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write('\n')


# ══════════════════════════════════════════════════════════════════════════════
#  POST-SESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
df      = pd.DataFrame(frame_records)
spmm_df = pd.DataFrame(spmm_records)

# ── Terminal summary ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FULL SYSTEM BENCHMARK — {NUM_TEST_SAMPLE} frames")
print(f"{'='*65}")

print(f"\n  Stage timing (mean):")
print(f"  {'Stage':<30} {'Mean (ms)':>10}")
print(f"  {'─'*42}")
for col, label in [
    ('t_data_loading_ms',       'Data Loading'),
    ('t_graph_construction_ms', 'Graph Construction'),
    ('t_gnn_ms',                'GNN Inference'),
    ('t_post_ms',               'Post-processing'),
    ('t_end_to_end_ms',         'End-to-End'),
]:
    print(f"  {label:<30} {df[col].mean():>10.1f}")

print(f"\n  Graph Construction sub-stages (mean):")
for col, label in [
    ('gc_frnn_ms',       'FRNN search  (sparse)'),
    ('gc_edge_build_ms', 'Edge build   (memory)'),
    ('gc_downsample_ms', 'Voxel sample (dense)'),
]:
    print(f"  {label:<28} {df[col].mean():>10.1f}")

print(f"\n  GNN Kernel Classification (mean % across all frames):")
print(f"  {'Category':<15} {'Mean %':>10} {'Mean ms':>10}")
print(f"  {'─'*37}")
for cat in TAXONOMY:
    print(f"  {cat:<15} {df[f'gnn_pct_{cat}'].mean():>9.1f}% "
          f"{df[f'gnn_{cat}_ms'].mean():>10.1f}")

all_match = spmm_df['coo_match'].all() and spmm_df['csr_match'].all()
print(f"\n  SpMM numerical match all frames: "
      f"{'YES ✓' if all_match else 'FAILURES ✗'}")
print(f"  SpMM mean timing (feat_dim={FEAT_DIM}, {N_RUNS} runs/frame):")
print(f"    tf.gather : {spmm_df['gather_mean'].mean():.3f} ms")
print(f"    SpMM COO  : {spmm_df['coo_mean'].mean():.3f} ms")
print(f"    SpMM CSR  : {spmm_df['csr_mean'].mean():.3f} ms")

print(f"\n  BOTTLENECK RANKING:")
total_ms = df['t_end_to_end_ms'].mean()
for rank, (label, ms) in enumerate(sorted([
    ('Data Loading',       df['t_data_loading_ms'].mean()),
    ('Graph Construction', df['t_graph_construction_ms'].mean()),
    ('GNN Inference',      df['t_gnn_ms'].mean()),
    ('Post-processing',    df['t_post_ms'].mean()),
], key=lambda x: x[1], reverse=True), 1):
    print(f"    #{rank}  {label:<25} {ms:>8.1f} ms "
          f"({ms/total_ms*100:.1f}%)")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
df.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_full.csv'),  index=False)
spmm_df.to_csv(os.path.join(OUTPUT_DIR, 'benchmark_spmm.csv'), index=False)
print(f"\n  CSVs → {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — GNN Kernel Classification (mean bar only)
# ══════════════════════════════════════════════════════════════════════════════
cats = list(TAXONOMY.keys())

fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.suptitle(
    'GNN Inference — Kernel Classification\n'
    f'Mean across {NUM_TEST_SAMPLE} frames | FULL_TRACE every frame',
    fontsize=13, fontweight='bold')

mean_pcts = [df[f'gnn_pct_{cat}'].mean() for cat in cats]
bars1 = ax1.bar(range(len(cats)), mean_pcts,
                color=[TAXONOMY_COLORS[c] for c in cats],
                edgecolor='black', linewidth=0.8, width=0.5)
for bar, pct in zip(bars1, mean_pcts):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{pct:.1f}%',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(cats)))
ax1.set_xticklabels([TAXONOMY_LABELS[c] for c in cats], fontsize=10)
ax1.set_ylabel('Mean % of kernel time', fontsize=12)
ax1.set_ylim(0, max(mean_pcts) * 1.25)
ax1.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, 'gnn_kernel_classification.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot 1 → {p1}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Full System Classification (mean bar only)
# ══════════════════════════════════════════════════════════════════════════════
sys_sparse = df['gnn_sparse_ms'] + df['gc_sparse_ms']
sys_dense  = df['gnn_dense_ms']  + df['gc_dense_ms']  + df['post_dense_ms']
sys_memory = (df['gnn_memory_ms'] + df['gc_memory_ms']
              + df['post_memory_ms'] + df['t_data_loading_ms'])
sys_other  = df['gnn_other_ms']  + df['post_other_ms']
sys_total  = sys_sparse + sys_dense + sys_memory + sys_other

sys_mean_pcts = [
    (sys_sparse / sys_total * 100).mean(),
    (sys_dense  / sys_total * 100).mean(),
    (sys_memory / sys_total * 100).mean(),
    (sys_other  / sys_total * 100).mean(),
]

fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.suptitle(
    'Full Pipeline Classification\n'
    f'Mean across {NUM_TEST_SAMPLE} frames | '
    'Graph Construction + GNN + Post-processing',
    fontsize=13, fontweight='bold')

bars2 = ax2.bar(range(4), sys_mean_pcts,
                color=[TAXONOMY_COLORS[c] for c in cats],
                edgecolor='black', linewidth=0.8, width=0.5)
for bar, pct in zip(bars2, sys_mean_pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{pct:.1f}%',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels([TAXONOMY_LABELS[c] for c in cats], fontsize=10)
ax2.set_ylabel('Mean % of pipeline time', fontsize=12)
ax2.set_ylim(0, max(sys_mean_pcts) * 1.25)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, 'system_classification.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot 2 → {p2}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — SpMM Comparison (mean bar only)
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 6))
fig3.suptitle(
    'SpMM Implementation Comparison\n'
    f'Mean across {NUM_TEST_SAMPLE} frames | feat_dim={FEAT_DIM} | '
    f'{"All frames match ✓" if all_match else "Mismatch ✗"}',
    fontsize=13, fontweight='bold')

means3  = [spmm_df['gather_mean'].mean(),
           spmm_df['coo_mean'].mean(),
           spmm_df['csr_mean'].mean()]
colors3 = ['#2196F3', '#FF8C00', '#4CAF50']
labels3 = ['tf.gather\n(TF kernel)',
           'SpMM COO\n(numpy)',
           'SpMM CSR\n(scipy)']

bars3 = ax3.bar(range(3), means3,
                color=colors3,
                edgecolor='black', linewidth=0.7, width=0.45)
for bar, mean in zip(bars3, means3):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(means3) * 0.02,
             f'{mean:.1f} ms',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(labels3, fontsize=10)
ax3.set_ylabel('Mean time (ms)', fontsize=12)
ax3.set_ylim(0, max(means3) * 1.25)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, 'spmm_comparison.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot 3 → {p3}")

print("\nDone.\n")