import os, glob
import numpy as np
import matplotlib.pyplot as plt


def plot_corr_curves():
    res_folder_1 = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/corrnet"
    pred_corr_filelist = glob.glob(os.path.join(res_folder_1, "*_nnind.npy"))
    ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    acc_all = np.zeros(len(ratio_list))
    for pred_corr_filename in pred_corr_filelist:
        gt_corr_filename = pred_corr_filename.replace("_nnind.npy", "_gt_corr.npy")
        pts_filename = pred_corr_filename.replace("_nnind.npy", "_pts.npy")
        pts = np.load(pts_filename)
        pred_corr = np.load(pred_corr_filename)
        gt_corr = np.load(gt_corr_filename)
        pred_pos = pts[pred_corr[gt_corr[:, 0]]]
        gt_pos = pts[gt_corr[:, 1]]
        dist = np.sqrt(np.sum((pred_pos - gt_pos) ** 2, axis=-1))
        for i, each_ratio in enumerate(ratio_list):
            acc = (dist < each_ratio).sum() / len(gt_corr)
            b_acc = acc * 100.0
            acc_all[i] += b_acc
    acc_all /= len(pred_corr_filelist)
    plt.plot(ratio_list, acc_all, 'c-', label="CorrNet")
    # plt.plot(t, ours_kmeans_init_adjbone, 'g-', label="skinning_adjbone")
    # plt.plot(t, ours_cpu, 'b--', label="cpu")
    # plt.plot(t, ours_gpu, 'm--', label="gpu")
    plt.legend(loc="upper left")
    plt.xlabel("Tolerances")
    plt.ylabel("Corr. Acc (%)")
    plt.show()
    return


if __name__ == "__main__":
    plot_corr_curves()
