import numpy as np
import open3d as o3d
from PIL import Image
import pycolmap as pcm

from pc_render import render_bunny
from utils import homogeneous, skew_lines_nearest_point
#from img_match import matching_points
from img_match_loftr import matching_points_LoFTR as matching_points
import cv2 as cv
import time

def reconstruct_3d(points1, points2, img_name1, img_name2, resize_ratio=1.0):
    N = len(points1)
    cam1 = pcm.infer_camera_from_image(img_name1)
    cam2 = pcm.infer_camera_from_image(img_name2)
    #print('cam1', cam1.summary())
    #print('cam2', cam2.summary())

    # if the img was resized (upsampled or downsampled),
    # then effective focal length should be changed
    cam1.focal_length /= resize_ratio
    cam2.focal_length /= resize_ratio

    matches = np.stack([np.arange(N)]*2, axis=1)
    options_2v = pcm.TwoViewGeometryOptions()
    options_2v.compute_relative_pose = True
    options_2v.min_num_inliers = 8
    g = pcm.estimate_calibrated_two_view_geometry(cam1, points1, cam2, points2, matches, options_2v)
    #print("E,F,H", g.E, g.F, g.H)
    #print("2f1", g.cam2_from_cam1.summary())
    tik = time.time()
    pc = triangulate_my(cam1, cam2, points1, points2, g)
    #pc = triangulate_cv(cam1, cam2, points1, points2, g)
    tok = time.time()
    print('time cost:', tok - tik)

    pc = np.array(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    return

def reconstruct_3d_cv(points1, points2, img_name1, img_name2, resize_ratio=1.0):
    cam1 = pcm.infer_camera_from_image(img_name1)
    cam2 = pcm.infer_camera_from_image(img_name2)
    cam1.focal_length /= resize_ratio
    cam2.focal_length /= resize_ratio
    points1, points2 = matching_points(img_name1, img_name2)
    K = cam1.calibration_matrix()
    E, mask = cv.findEssentialMat(points1, points2, K)
    inliers1, inliers2 = points1[mask], points2[mask]
    _, R, t, _ = cv.recoverPose(E, inliers1, inliers2, K)
    Rt = np.hstack([R, t])
    P1 = cam1.calibration_matrix() @ np.eye(3, 4)
    P2 = cam1.calibration_matrix() @ Rt
    p4d = cv.triangulatePoints(P1, P2, points1.T, points2.T)
    pc = (p4d[:3]/p4d[3]).T
    pc = np.array(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    return

def triangulate_cv(cam1, cam2, points1, points2, g):
    # cam1: pycolmap.Camera
    # cam2: pycolmap.Camera
    # points1: [N, 2]
    # points2: [N, 2]
    # g: pycolmap.TwoViewGeometry
    points1 = np.array(points1)
    points2 = np.array(points2)

    # solution0 (using opencv):
    P1 = cam1.calibration_matrix() @ np.eye(3, 4)
    P2 = cam2.calibration_matrix() @ g.cam2_from_cam1.matrix()
    p4d = cv.triangulatePoints(P1, P2, points1.T, points2.T)
    pc = (p4d[:3]/p4d[3]).T
    return pc

def triangulate_my(cam1, cam2, points1, points2, g):
    # cam1: pycolmap.Camera
    # cam2: pycolmap.Camera
    # points1: [N, 2]
    # points2: [N, 2]
    # g: pycolmap.TwoViewGeometry

    # np.linalg.inv(cam1.calibration_matrix()) @ homogeneous(px1) == homogeneous(cam1.cam_from_img(px1))

    points1 = np.array(points1)
    points2 = np.array(points2)

    # solution1 (using proj matrix):
    P1 = cam1.calibration_matrix() @ np.hstack([np.eye(3), np.zeros([3,1])])
    P2 = cam2.calibration_matrix() @ g.cam2_from_cam1.matrix()
    R1 = P1[:3,:3]
    t1 = P1[:3,3]
    R2 = P2[:3,:3]
    t2 = P2[:3,3]
    d1 = (np.linalg.pinv(R1) @ homogeneous(points1).T).T
    d2 = (np.linalg.pinv(R2) @ homogeneous(points2).T).T
    o1 = np.reshape(-np.linalg.pinv(R1) @ t1, [1, 3])
    o2 = np.reshape(-np.linalg.pinv(R2) @ t2, [1, 3])

    ## solution2 (using cam 1 coord):
    #R2f1 = g.cam2_from_cam1.matrix()[:3,:3]
    #p1c1 =  homogeneous(cam1.cam_from_img(points1))
    #p2c2 =  homogeneous(cam2.cam_from_img(points2))
    #d1 = p1c1
    #d2 = (np.linalg.inv(R2f1) @ p2c2.T).T
    #o1 = np.zeros([1, 3])
    #o2 = np.reshape(-R2f1.T @ g.cam2_from_cam1.translation, [1, 3])

    ## solution3 (using cam 2 coord):
    #R2f1 = g.cam2_from_cam1.matrix()[:3,:3]
    #p1c1 =  homogeneous(cam1.cam_from_img(points1))
    #p2c2 =  homogeneous(cam2.cam_from_img(points2))
    #d1 = (R2f1 @ p1c1.T).T
    #d2 = p2c2
    #o1 = np.reshape(g.cam2_from_cam1.translation, [1, 3])
    #o2 = np.zeros([1, 3])

    pc, min_d, valid = skew_lines_nearest_point(o1, d1, o2, d2)
    pc = pc[valid.flatten() == True]
    return pc

if __name__ == '__main__':
    #bunny_np, points1, points2 = render_bunny()
    #reconstruct_3d(points1, points2, 'bunny1.png', 'bunny2.png')

    img_name1 = 'test111.jpg'
    img_name2 = 'test222.jpg'
    points1, points2 = matching_points(img_name1, img_name2)
    print('points1, points2')
    print(points1.shape, points2.shape)
    print(points1[:3], points2[:3])
    reconstruct_3d_cv(points1, points2, img_name1, img_name2, resize_ratio=1/4)
