import os, glob, numpy as np
from tqdm import tqdm

# 0114: 0.06631005450531288
# 0124: 0.06351847700222775
def eval_flow():
    res_folder = "/mnt/neghvar/mnt/DATA_LINUX/zhan/output/deformnet_modelsresources/"
    pred_flow_filelist = glob.glob(os.path.join(res_folder, "*_1_pred_flow.npy"))
    diff = []
    model_ids = []
    for pred_flow_filename in tqdm(pred_flow_filelist):
        model_ids.append(int(pred_flow_filename.split("/")[-1].split("_")[0]))
        diff_t = []
        for t in np.arange(1, 6):
            pred_flow_filename_t = pred_flow_filename.replace("_1_pred_flow.npy", f"_{t}_pred_flow.npy")
            pred_flow_t = np.load(pred_flow_filename_t)
            gt_flow_t = np.load(pred_flow_filename_t.replace("_pred_flow.npy", "_gt_flow.npy"))
            diff_t.append(np.mean(np.sqrt(np.sum((gt_flow_t - pred_flow_t)**2, axis=1))))
        diff.append(np.asarray(diff_t).mean())
    diff = np.asarray(diff)
    #print(diff.mean())
    model_ids = np.asarray(model_ids)
    sorted_ids = np.argsort(diff)
    model_ids_sorted = model_ids[sorted_ids]
    print(model_ids_sorted)


if __name__ == "__main__":
    eval_flow()
