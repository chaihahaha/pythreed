import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap as pcm

from pc_render import render_bunny
from utils import homogeneous, skew_lines_nearest_point

def reconstruct_3d(points1, points2, img_name1, img_name2):
    N = len(points1)
    cam1 = pcm.infer_camera_from_image(img_name1)
    cam2 = pcm.infer_camera_from_image(img_name2)
    matches = np.stack([np.arange(N)]*2, axis=1)
    options_2v = pcm.TwoViewGeometryOptions()
    options_2v.compute_relative_pose = True
    options_2v.min_num_inliers = 8
    g = pcm.estimate_two_view_geometry(cam1, points1, cam2, points2, matches, options_2v)
    pc = []
    for px1, px2 in zip(points1, points2):
        p3d = triangulate(cam1, cam2, px1, px2, g)
        pc.append(p3d)
    pc = np.array(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    return

def triangulate(cam1, cam2, px1, px2, g):
    p1c1 = np.linalg.inv(cam1.calibration_matrix()) @ homogeneous(cam1.cam_from_img(px1))
    p2c2 = np.linalg.inv(cam2.calibration_matrix()) @ homogeneous(cam2.cam_from_img(px2))
    p1w = g.cam2_from_cam1.matrix() @ homogeneous(p1c1)
    p2w = p2c2
    d1 = p1w + g.cam2_from_cam1.translation
    d2 = p2w
    p3d = skew_lines_nearest_point(p1w, d1, p2w, d2)
    return p3d

if __name__ == '__main__':
    bunny_np, points1, points2 = render_bunny()
    reconstruct_3d(points1, points2, 'bunny1.png','bunny2.png')
