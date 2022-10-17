import numpy as np
import open3d as o3d
import copy
from utils.colormaps import *


def drawSphere(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def drawCone(bottom_center, top_position, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.005, height=np.linalg.norm(top_position - bottom_center)+1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4: # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    #print(R)
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone


def show_obj_rig(mesh, rig):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    parent_id = rig.names.index(rig.root_name)
    vis.add_geometry(drawSphere(rig.pos[parent_id], 0.007, color=[0.1, 0.1, 0.1]))
    parents_list = [parent_id]
    while parents_list:
        parents_list_next = []
        for j in range(len(rig.pos)):
            if rig.hierarchy[j] in parents_list:
                parent_id = rig.hierarchy[j]
                vis.add_geometry(drawSphere(rig.pos[j], 0.007, color=[1.0, 0.0, 0.0]))
                vis.add_geometry(drawCone(rig.pos[parent_id], rig.pos[j]))
                parents_list_next.append(j)
        parents_list = parents_list_next

    #param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    #ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    #vis.poll_events()
    #vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def visualize_flow_seq(pred_traj, gt_pts=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for t in range(pred_traj.shape[1] // 3):
        pcd_v = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pred_traj[:, 3 * t:3 * t + 3]))
        pcd_v.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(pcd_v.translate([t * 1.0, 0.0, 0.0]))
        if gt_pts is not None:
            pcd_p = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(gt_pts[:, 3 * t:3 * t + 3]))
            pcd_p.paint_uniform_color([0.0, 0.0, 1.0])
            vis.add_geometry(pcd_p.translate([t * 1.0, 0.0, 0.0]))
    vis.run()
    # vis.poll_events()
    # vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def visualize_corr_vismask(vtx, pts, corr, vismask=None):
    pcd_vtx = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx))
    pts += np.array([[1.0, 0.0, 0.0]])
    pcd_pts = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    
    pcd_vtx.paint_uniform_color([0.1, 0.0, 0.0])
    pcd_pts.paint_uniform_color([0.0, 0.0, 1.0])

    corr_1 = vtx[corr[:, 0]]
    corr_2 = pts[corr[:, 1]]
    corr_pts = np.concatenate((corr_1, corr_2), axis=0)
    corr_lines = np.array([[i, i + len(corr_1)] for i in range(len(corr_1))])
    corr_line_set = o3d.geometry.LineSet()
    corr_line_set.points = o3d.utility.Vector3dVector(corr_pts)
    corr_line_set.lines = o3d.utility.Vector2iVector(corr_lines)
    colors = [[0.7, 0.7, 0.7] for i in range(len(corr_lines))]
    corr_line_set.colors = o3d.utility.Vector3dVector(colors)

    if vismask is not None:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('jet')
        pcd_vtx_vismask = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx))
        pcd_vtx_vismask_colors = cmap(vismask)[:,0:3]
        pcd_vtx_vismask.colors=o3d.utility.Vector3dVector(pcd_vtx_vismask_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_vtx)
    vis.add_geometry(pcd_pts)
    vis.add_geometry(corr_line_set)
    if vismask is not None:
        vis.add_geometry(pcd_vtx_vismask.translate([2.0, 0.0, 0.0]))
    vis.run()
    vis.destroy_window()

def visualize_seg(pts, nidx, mesh=None):
    cmap = create_ade20k_label_colormap()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cmap[nidx] / 255.0)
    vis.add_geometry(pcd)
    if mesh is not None:
        mesh_new = copy.deepcopy(mesh)
        mesh_new.vertices = o3d.utility.Vector3dVector(pts)
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_new)
        mesh_ls.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(mesh_ls)
    vis.run()
    #vis.poll_events()
    #vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def visualize_seg_skel(mesh, label, skel):
    new_cmap = create_ade20k_label_colormap()
    color_pred = new_cmap[label]/ 255.0
    pcd = o3d.geometry.PointCloud(points=mesh.vertices)
    pcd.colors = o3d.utility.Vector3dVector(color_pred)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(drawSphere(skel.pos[skel.root_id], radius=0.007, color=[0.0, 0.0, 0.0]))
    this_level = [skel.root_id]
    while this_level:
        next_level = []
        for p_id in this_level:
            ch_list = np.argwhere(skel.hierarchy == p_id).squeeze(axis=1)
            for ch_id in ch_list:
                vis.add_geometry(drawSphere(skel.pos[ch_id], radius=0.007, color=[1.0, 0.0, 0.0]))
                vis.add_geometry(drawCone(skel.pos[p_id], skel.pos[ch_id]))
            next_level += ch_list.tolist()
        this_level = next_level
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(mesh_ls)
    vis.run()
    #vis.poll_events()
    #vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def visualize_seg_joints(vtx, labels, joint_pos, mesh=None):
    cmap = create_ade20k_label_colormap()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_v = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx))
    color_pred = cmap[labels] / 255.0
    pcd_v.colors = o3d.utility.Vector3dVector(color_pred)
    vis.add_geometry(pcd_v)
    for j in range(len(joint_pos)):
        vis.add_geometry(drawSphere(joint_pos[j], radius=0.007, color=cmap[j] / 255.0))
    if mesh is not None:
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_ls.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(mesh_ls)
    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def volume_to_cubes(volume, threshold=0, dim=[1., 1., 1.]):
    #o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    o = np.array([0, 0, 0])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    lines = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]

                    points.append(np.array([xx, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, yy, ZZ])[np.newaxis, :])
                    points.append(np.array([xx, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, yy, ZZ])[np.newaxis, :])

                    lines.append(np.array([vidx + 1, vidx + 2]))
                    lines.append(np.array([vidx + 2, vidx + 6]))
                    lines.append(np.array([vidx + 6, vidx + 5]))
                    lines.append(np.array([vidx + 1, vidx + 5]))

                    lines.append(np.array([vidx + 1, vidx + 1]))
                    lines.append(np.array([vidx + 3, vidx + 3]))
                    lines.append(np.array([vidx + 7, vidx + 7]))
                    lines.append(np.array([vidx + 5, vidx + 5]))

                    lines.append(np.array([vidx + 0, vidx + 3]))
                    lines.append(np.array([vidx + 0, vidx + 4]))
                    lines.append(np.array([vidx + 4, vidx + 7]))
                    lines.append(np.array([vidx + 7, vidx + 3]))

    return points, lines


def show_mesh_vox(mesh_filename, vox):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vox_pts, vox_lines = volume_to_cubes(vox.data)
    vox_pts = np.concatenate(vox_pts, axis=0)
    line_set_vox = o3d.geometry.LineSet()
    line_set_vox.points = o3d.utility.Vector3dVector(vox_pts+np.array(vox.translate)[np.newaxis, :])
    line_set_vox.lines = o3d.utility.Vector2iVector(vox_lines)
    colors = [[0.0, 0.0, 1.0] for i in range(len(vox_lines))]
    line_set_vox.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set_vox)

    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.run()
    vis.destroy_window()

    return

def visualize_track(mesh_0, vert_gt, vert_pred, pts_gt=None):
    if pts_gt is not None:
        pcd_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_gt.astype(np.float32)))
        pcd_pts.paint_uniform_color([0.0, 0.0, 1.0])
    mesh_update = copy.deepcopy(mesh_0)
    mesh_update.vertices = o3d.utility.Vector3dVector(vert_pred)
    mesh_gt = copy.deepcopy(mesh_0)
    mesh_gt.vertices = o3d.utility.Vector3dVector(vert_gt)
    mesh_update_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_update)
    mesh_update_ls.paint_uniform_color([1.0, 0.0, 0.0])
    mesh_gt_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_gt)
    mesh_gt_ls.paint_uniform_color([0.0, 0.0, 1.0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_update_ls)
    vis.add_geometry(mesh_gt_ls)
    if pts_gt is not None:
        vis.add_geometry(pcd_pts.translate([1.0, 0.0, 0.0]))
    #vis.run()
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def draw_shifted_pts(mesh, pts, weights=None):
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('YlOrRd')
        #weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        #weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    #param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    #ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image

def draw_joints(mesh, pts, gt_joint=None):
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    for joint_pos in pts:
        vis.add_geometry(drawSphere(joint_pos, 0.006, color=[1.0, 0.0, 0.0]))

    if gt_joint is not None:
        for joint_pos in gt_joint:
            vis.add_geometry(drawSphere(joint_pos, 0.006, color=[0.0, 0.0, 1.0]))

    #param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    #ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image