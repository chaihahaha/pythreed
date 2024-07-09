import numpy as np
import open3d as o3d
from PIL import Image
from utils import compute_intrinsic_matrix, compute_extrinsic_matrix, direction_to_quaternion

def render_point_cloud_to_binary_image(K, Rt, P, width, height):
    """
    Compute the binary image using the model view projection method.
    
    Parameters:
    K (numpy.ndarray): Intrinsic matrix of shape (3, 3).
    R (numpy.ndarray): Rotation matrix of shape (3, 3).
    t (numpy.ndarray): Translation vector of shape (3,).
    P (numpy.ndarray): 3D point cloud of shape (N, 3).
    width (int): Width of the output image.
    height (int): Height of the output image.
    
    Returns:
    numpy.ndarray: Binary image of shape (height, width).
    """
    # Step 1: Transform 3D points to camera coordinates
    P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))
    print('P_homogeneous', P_homogeneous)
    P_camera = Rt @ P_homogeneous.T
    print('P_camera', P_camera)
    
    # Step 2: Project to 2D image coordinates
    P_image_homogeneous = K @ P_camera
    print('P_image_homogeneous', P_image_homogeneous)
    
    # Convert to Cartesian coordinates
    P_image = P_image_homogeneous[:2, :] / P_image_homogeneous[2, :]
    print('P_image', P_image)
    
    # Step 3: Create the binary image
    binary_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(P_image.shape[1]):
        u, v = P_image[:, i]
        u = int(np.round(u))
        v = int(np.round(v))
        if 0 <= u < width and 0 <= v < height:
            binary_image[v, u] = 1
    
    return binary_image, P_image.T


def load_bunny_np_download():
    bunny = o3d.data.BunnyMesh()
    bunny_np = np.array(bunny.points)
    return bunny_np

def load_bunny_np():
    bunny = o3d.io.read_point_cloud('BunnyMesh.ply')
    bunny_np = np.array(bunny.points)
    return bunny_np

def render_bunny():
    bunny_np = load_bunny_np()
    #print('mean',np.mean(bunny_np, axis=0))
    #print('max',np.max(bunny_np, axis=0))

    K1 = compute_intrinsic_matrix(200,200,256,256,0)
    direction1 = np.array([-1, -1, -1])
    q1 = direction_to_quaternion(direction1)
    t1 = np.array([0.1, 0., 0.1])
    Rt1 = compute_extrinsic_matrix(q1, t1)
    img1, pimg1 = render_point_cloud_to_binary_image(K1, Rt1, bunny_np, 512, 512)
    #plt.imshow(img1, cmap='gray')
    #plt.savefig('bunny1.png')
    img1_pil = Image.fromarray(img1*255).convert('L')
    img1_pil.save('bunny1.png')


    K2 = compute_intrinsic_matrix(250,250,256,256,0)
    direction2 = np.array([-0, -0, -1])
    q2 = direction_to_quaternion(direction2)
    t2 = np.array([0.0, 0.1, 0.2])
    Rt2 = compute_extrinsic_matrix(q2, t2)
    img2, pimg2 = render_point_cloud_to_binary_image(K2, Rt2, bunny_np, 512, 512)
    img2_pil = Image.fromarray(img2*255).convert('L')
    img2_pil.save('bunny2.png')
    return bunny_np, pimg1, pimg2

