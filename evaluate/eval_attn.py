import glob, os, numpy as np, open3d as o3d
import matplotlib.pyplot as plt


if __name__ == "__main__":
    pred_attn_ours_folder = "/media/zhanxu/4T/motionAnim_results/ModelsResource_comparison/ours/attn/"
    pred_attn_rignet_folder = "/media/zhanxu/4T/motionAnim_results/ModelsResource_comparison/rignet/attn/"
    pred_attn_gt_flow_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/masknet_motion_gt_flow/"
    gt_attn_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/ModelResource_RigNetv1_preproccessed/pretrain_attention/"

    pred_attn_ours_filelist = glob.glob(os.path.join(pred_attn_ours_folder, f"*_attn.npy"))
    prec_all_ours, rec_all_ours, prec_all_rignet, rec_all_rignet, prec_all_gt_flow, rec_all_gt_flow = \
        np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

    for pred_attn_ours_filename in pred_attn_ours_filelist:
        model_id = pred_attn_ours_filename.split("/")[-1].split("_")[0]
        gt_attn_filename = os.path.join(gt_attn_folder, f"{model_id}.txt")
        gt_attn = np.loadtxt(gt_attn_filename)

        pred_attn_ours = np.load(pred_attn_ours_filename).squeeze(axis=1)
        pred_attn_ours = (pred_attn_ours - np.min(pred_attn_ours)) / (np.max(pred_attn_ours) - np.min(pred_attn_ours))

        pred_attn_rignet_filename = os.path.join(pred_attn_rignet_folder, f"{model_id}_attn.npy")
        pred_attn_rignet = np.load(pred_attn_rignet_filename).squeeze(axis=1)
        pred_attn_rignet = (pred_attn_rignet - np.min(pred_attn_rignet)) / (np.max(pred_attn_rignet) - np.min(pred_attn_rignet))

        pred_attn_gt_flow_filename = os.path.join(pred_attn_gt_flow_folder, f"{model_id}_attn.npy")
        pred_attn_gt_flow = np.load(pred_attn_gt_flow_filename).squeeze(axis=1)
        pred_attn_gt_flow = (pred_attn_gt_flow - np.min(pred_attn_gt_flow)) / (np.max(pred_attn_gt_flow) - np.min(pred_attn_gt_flow))

        for i, t in enumerate(np.arange(0.0, 1.0, 0.1)):
            pred_attn_ours_t = (pred_attn_ours > t)
            prec_ours_t = (pred_attn_ours_t & gt_attn.astype(np.bool)).sum() / (pred_attn_ours_t.sum() + 1e-6)
            rec_ours_t = (pred_attn_ours_t & gt_attn.astype(np.bool)).sum() / (gt_attn.sum() + 1e-6)
            prec_all_ours[i] += prec_ours_t
            rec_all_ours[i] += rec_ours_t

            pred_attn_rignet_t = (pred_attn_rignet > t)
            prec_rignet_t = (pred_attn_rignet_t & gt_attn.astype(np.bool)).sum() / (pred_attn_rignet_t.sum() + 1e-6)
            rec_rignet_t = (pred_attn_rignet_t & gt_attn.astype(np.bool)).sum() / (gt_attn.sum() + 1e-6)
            prec_all_rignet[i] += prec_rignet_t
            rec_all_rignet[i] += rec_rignet_t

            pred_attn_gt_flow_t = (pred_attn_gt_flow > t)
            prec_t_gt_flow = (pred_attn_gt_flow_t & gt_attn.astype(np.bool)).sum() / (pred_attn_gt_flow_t.sum() + 1e-6)
            rec_t_gt_flow = (pred_attn_gt_flow_t & gt_attn.astype(np.bool)).sum() / (gt_attn.sum() + 1e-6)
            prec_all_gt_flow[i] += prec_t_gt_flow
            rec_all_gt_flow[i] += rec_t_gt_flow

    prec_all_ours = prec_all_ours / len(pred_attn_ours_filelist)
    rec_all_ours = rec_all_ours / len(pred_attn_ours_filelist)
    prec_all_rignet = prec_all_rignet / len(pred_attn_ours_filelist)
    rec_all_rignet = rec_all_rignet / len(pred_attn_ours_filelist)
    prec_all_gt_flow = prec_all_gt_flow / len(pred_attn_ours_filelist)
    rec_all_gt_flow = rec_all_gt_flow / len(pred_attn_ours_filelist)

    plt.plot(rec_all_ours, prec_all_ours, "b-", label="ours")
    plt.plot(rec_all_rignet, prec_all_rignet, "r-", label="rignet")
    plt.plot(rec_all_gt_flow, prec_all_gt_flow, "g-", label="gt_flow")
    plt.title('Precision-recall curve for attention prediction')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc="upper right")
    plt.show()



