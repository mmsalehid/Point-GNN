"""Compatibility aliases for old and new Open3D Python APIs."""

import open3d as _open3d

if hasattr(_open3d, "geometry"):
    PointCloud = _open3d.geometry.PointCloud
    LineSet = _open3d.geometry.LineSet
    TriangleMesh = _open3d.geometry.TriangleMesh
    Vector3dVector = _open3d.utility.Vector3dVector
    Vector2iVector = _open3d.utility.Vector2iVector
    Visualizer = _open3d.visualization.Visualizer
    draw_geometries = _open3d.visualization.draw_geometries

    def voxel_down_sample(point_cloud, voxel_size):
        return point_cloud.voxel_down_sample(voxel_size)

    def create_mesh_cylinder(radius=1.0, height=2.0, resolution=20, split=4):
        return TriangleMesh.create_cylinder(
            radius=radius, height=height, resolution=resolution, split=split)
else:
    PointCloud = _open3d.PointCloud
    LineSet = _open3d.LineSet
    Vector3dVector = _open3d.Vector3dVector
    Vector2iVector = _open3d.Vector2iVector
    Visualizer = _open3d.Visualizer
    draw_geometries = _open3d.draw_geometries
    voxel_down_sample = _open3d.voxel_down_sample
    create_mesh_cylinder = _open3d.create_mesh_cylinder
