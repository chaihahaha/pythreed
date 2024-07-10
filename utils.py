import numpy as np
#import pycolmap as pcm


def compute_extrinsic_matrix(q, t):
    """
    Compute the extrinsic matrix [R|t] using a quaternion and a translation vector.
    
    Parameters:
    q (numpy.ndarray): Quaternion (q_w, q_x, q_y, q_z)
    t (numpy.ndarray): Translation vector (t_x, t_y, t_z)
    
    Returns:
    numpy.ndarray: Extrinsic matrix of shape (4, 4)
    """
    R = quaternion_to_rotation_matrix(q)
    Rt = np.zeros([3,4])
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt

def direction_to_quaternion(v):
    """
    Compute a quaternion representing the direction of a 3D vector.
    
    Parameters:
    v (numpy.ndarray): A 3D direction vector.
    
    Returns:
    numpy.ndarray: A quaternion (q_w, q_x, q_y, q_z)
    """
    # Normalize the direction vector
    v_normalized = v / np.linalg.norm(v)
    
    # Reference direction (z-axis)
    u = np.array([0, 0, 1])
    
    # Compute the axis of rotation (cross product)
    a = np.cross(u, v_normalized)
    
    # Compute the angle of rotation (dot product and arccos)
    cos_theta = np.dot(u, v_normalized)
    theta = np.arccos(cos_theta)
    
    # Handle the case when the direction is the same as the reference direction
    if np.isclose(cos_theta, 1.0):
        return np.array([1, 0, 0, 0])  # No rotation, identity quaternion
    
    # Handle the case when the direction is opposite to the reference direction
    if np.isclose(cos_theta, -1.0):
        return np.array([0, 1, 0, 0])  # 180-degree rotation around the x-axis
    
    # Normalize the axis of rotation
    a_normalized = a / np.linalg.norm(a)
    
    # Construct the quaternion
    q_w = np.cos(theta / 2)
    q_xyz = a_normalized * np.sin(theta / 2)
    
    q = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
    
    return q

def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def compute_intrinsic_matrix(fx, fy, cx, cy, s):
    # fx, fy: focal length
    # cx, cy: optical center, principal point, in pixels
    # s: skew coefficient, =0 when image axes are perpendicular
    K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
    K = np.array(K)
    return K

def homogeneous(a):
    return np.pad(a, [0,1], constant_values=1)

def skew_lines_nearest_point(p1, d1, p2, d2):
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    n = np.cross(d1, d2)
    n1 = np.cross(d1, n)
    n2 = np.cross(d2, n)
    c1 = p1 + np.dot(p2 - p1, n2) * d1 / np.dot(d1, n2)
    c2 = p2 + np.dot(p1 - p2, n1) * d2 / np.dot(d2, n1)
    return (c1+c2)/2

