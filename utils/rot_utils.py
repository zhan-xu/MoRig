import numpy as np
import math
import torch

def isRotationMatrix(R):
    Rt = np.transpose(R, axes=(0, 2, 1))
    shouldBeIdentity = np.matmul(Rt, R)
    I = np.identity(3, dtype=R.dtype)[None, ...]
    n = (np.linalg.norm((I - shouldBeIdentity).reshape(-1, 9), axis=-1)) < 1e-6
    return n.sum() == len(n)


def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = np.sqrt(np.sum(v**2, axis=1))# batch
    v_mag = np.maximum(v_mag, 1e-8)
    v_mag = v_mag.reshape(batch,1)
    v_mag = np.repeat(v_mag, v.shape[1], axis=1)
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v


def cross_product(u, v):
    # u, v batch*n
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = np.concatenate((i.reshape(batch,1), j.reshape(batch,1), k.reshape(batch,1)),1)#batch*3
    return out


def mat2continuous6d(R):
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)

def mat2continuous6d_torch(R):
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)

def continuous6d2mat(ortho6d): 
    # batchx6
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.reshape(-1,3,1)
    y = y.reshape(-1,3,1)
    z = z.reshape(-1,3,1)
    matrix = np.concatenate((x,y,z), 2) #batch*3*3
    return matrix


def eular2mat(theta):
    batch = theta.shape[0]
    cos_0 = np.cos(theta[:, 0])
    cos_1 = np.cos(theta[:, 1])
    cos_2 = np.cos(theta[:, 2])
    sin_0 = np.sin(theta[:, 0])
    sin_1 = np.sin(theta[:, 1])
    sin_2 = np.sin(theta[:, 2])
    R_x = np.eye(3)[None, ...].repeat(batch, axis=0)
    R_y = np.eye(3)[None, ...].repeat(batch, axis=0)
    R_z = np.eye(3)[None, ...].repeat(batch, axis=0)
    R_x[:, 1, 1] = cos_0
    R_x[:, 1, 2] = -sin_0
    R_x[:, 2, 1] = sin_0
    R_x[:, 2, 2] = cos_0
    R_y[:, 0, 0] = cos_1
    R_y[:, 0, 2] = sin_1
    R_y[:, 2, 0] = -sin_1
    R_y[:, 2, 2] = cos_1
    R_z[:, 0, 0] = cos_2
    R_z[:, 0, 1] = -sin_2
    R_z[:, 1, 0] = sin_2
    R_z[:, 1, 1] = cos_2
    R = np.matmul(R_z, np.matmul( R_y, R_x ))
    return R


def mat2eular(R) :
    batch = R.shape[0]
    assert(isRotationMatrix(R))
    sy = np.sqrt(R[:, 0, 0] * R[:, 0, 0] +  R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    R_singular = R[singular]
    R_nonsingular = R[1-singular]
    eulars = np.zeros((batch, 3))
    if len(R_nonsingular) > 0:
        x_nonsin = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        y_nonsin = np.arctan2(-R[:, 2,0], sy)
        z_nonsin = np.arctan2(R[:, 1,0], R[:, 0,0])
        eulars[np.argwhere(1-singular).squeeze(axis=1)] = np.stack((x_nonsin, y_nonsin, z_nonsin), axis=1)
    if len(R_singular) > 0:
        x_sin = np.arctan2(-R[:, 1, 2], R[:, 1, 1])
        y_sin = np.arctan2(-R[:, 2, 0], sy)
        z_sin = np.zeros(batch)
        eulars[np.argwhere(singular).squeeze(axis=1)] = np.stack((x_sin, y_sin, z_sin), axis=1)
    return eulars

def continuous6d2eular(ortho6d): 
    # batchx6
    mat = continuous6d2mat(ortho6d)
    eulars = mat2eular(mat)
    return eulars

if __name__ == '__main__':
    #numeric value between -π and π
    alpha = 3.1
    beta = -1.2
    gamma = -3.1
    mat = eular2mat(np.array([[alpha, beta, gamma], [1.0, -0.5, 1.2]]))
    con6d = mat2continuous6d(mat)
    mat2 = continuous6d2mat(con6d)
    eulars = mat2eular(mat2)
    print(eulars)

