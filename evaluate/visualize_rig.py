import os, glob, open3d as o3d, cv2, numpy as np
from utils.vis_utils import show_obj_rig, drawSphere
from utils.rig_parser import Rig


def show_rigs():
    rig_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/test/"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/obj_remesh/"
    output_folder = "/media/zhanxu/4T/motionAnim_results/ModelsResource_comparison/visualization/"

    rig_filelist = glob.glob(os.path.join(rig_folder, "*_rig.txt"))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for rig_filename in rig_filelist:
        model_id = rig_filename.split("/")[-1].split("_")[0]
        rig = Rig(rig_filename)
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))
        img = show_obj_rig(mesh, rig)
        cv2.imwrite(os.path.join(output_folder, f"{model_id}_gt_rig.png"), img[:, 400:-400,::-1])

def show_joint_inner(mesh, joints):
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.paint_uniform_color([0.8, 0.8, 0.8])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    # draw mesh
    vis.add_geometry(mesh_ls)
    for joint in joints:
        vis.add_geometry(drawSphere(joint, 0.007, color=[1.0, 0.0, 0.0]))

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    image = vis.capture_screen_float_buffer()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    vis.destroy_window()
    return image


def show_joints():
    joint_folder = "/media/zhanxu/4T/motionAnim_results/ModelsResource_comparison/ours/res/"
    mesh_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/obj_remesh/"
    output_folder = "/media/zhanxu/4T/motionAnim_results/ModelsResource_comparison/visualization/"
    joint_filelist = glob.glob(os.path.join(joint_folder, "*_joint.npy"))

    for joint_filename in joint_filelist:
        model_id = joint_filename.split("/")[-1].split("_")[0]
        print(model_id)
        joints = np.load(joint_filename)
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_folder, f"{model_id}.obj"))
        image = show_joint_inner(mesh, joints)
        cv2.imwrite(os.path.join(output_folder, f"{model_id}_ours_joint.png"), image[:, 400:-400,::-1])


if __name__ == "__main__":
    #show_rigs()
    show_joints()