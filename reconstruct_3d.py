import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap as pcm

from pc_render import render_bunny
from utils import homogeneous, skew_lines_nearest_point
from img_match import matching_points

def reconstruct_3d(points1, points2, img_name1, img_name2):
    N = len(points1)
    cam1 = pcm.infer_camera_from_image(img_name1)
    cam2 = pcm.infer_camera_from_image(img_name2)
    #cam1.focal_length = 200
    #cam2.focal_length = 250
    matches = np.stack([np.arange(N)]*2, axis=1)
    options_2v = pcm.TwoViewGeometryOptions()
    options_2v.compute_relative_pose = True
    options_2v.min_num_inliers = 8
    g = pcm.estimate_calibrated_two_view_geometry(cam1, points1, cam2, points2, matches, options_2v)
    #g.cam2_from_cam1.translation = np.array([-0.1,0.1,0.1])
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
    # solution1 (using proj matrix):
    #P1 = cam1.calibration_matrix() @ np.hstack([np.eye(3), np.zeros([3,1])])
    #P2 = cam2.calibration_matrix() @ g.cam2_from_cam1.matrix()
    #R1 = P1[:3,:3]
    #t1 = P1[:3,3]
    #R2 = P2[:3,:3]
    #t2 = P2[:3,3]
    #d1 = np.linalg.pinv(R1) @ homogeneous(px1)
    #d2 = np.linalg.pinv(R2) @ homogeneous(px2)
    #o1 = -np.linalg.pinv(R1) @ t1
    #o2 = -np.linalg.pinv(R2) @ t2

    # solution2 (using cam 1 coord):
    ##p1c1 = np.linalg.inv(cam1.calibration_matrix()) @ homogeneous(px1)
    ##p2c2 = np.linalg.inv(cam2.calibration_matrix()) @ homogeneous(px2)

    #R2f1 = g.cam2_from_cam1.matrix()[:3,:3]
    #p1c1 =  homogeneous(cam1.cam_from_img(px1))
    #p2c2 =  homogeneous(cam2.cam_from_img(px2))
    #d1 = p1c1
    #d2 = np.linalg.inv(R2f1) @ p2c2
    #o1 = np.zeros(3)
    #o2 = -np.linalg.inv(R2f1) @ g.cam2_from_cam1.translation

    # solution3 (using cam 2 coord):
    R2f1 = g.cam2_from_cam1.matrix()[:3,:3]
    p1c1 =  homogeneous(cam1.cam_from_img(px1))
    p2c2 =  homogeneous(cam2.cam_from_img(px2))
    d1 = R2f1 @ p1c1
    d2 = p2c2
    o1 = g.cam2_from_cam1.translation
    o2 = np.zeros(3)

    p3d = skew_lines_nearest_point(o1, d1, o2, d2)
    return p3d

if __name__ == '__main__':
    #bunny_np, points1, points2 = render_bunny()
    #reconstruct_3d(points1, points2, 'bunny1.png', 'bunny2.png')

    img_name1 = 'test1.jpg'
    img_name2 = 'test2.jpg'
    points1, points2 = matching_points(img_name1, img_name2)
    reconstruct_3d(points1, points2, img_name1, img_name2)
