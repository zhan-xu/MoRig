import numpy as np
class KernelKMeans():
    def __init__(self, n_clusters=20, max_iter=100, w_euc=0.2, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers_euc = None
        self.centers_emb = None
        self.w_euc = w_euc
        self.tol = tol

    def calc_dist_emb(self, X, Y=None):
        if Y is None:
            return np.maximum(1.0 - np.matmul(X, X.T), 0)
        else:
            return np.maximum(1.0 - np.matmul(X, Y.T), 0)

    def calc_dist_euc(self, X, Y=None):
        if Y is None:
            return np.sqrt(np.sum((X[:, None, :] - X[None, ...]) ** 2, axis=-1))
        else:
            return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))

    def calc_dist(self, X_emb_all, X_euc_all, X_emb_sample, X_euc_sample):
        dist_euc = self.calc_dist_euc(X_euc_all, X_euc_sample)
        dist_emb = self.calc_dist_emb(X_emb_all, X_emb_sample)
        dist = dist_euc * self.w_euc + dist_emb / 2
        return dist

    def fps_embedding(self, pts):
        calc_distances = lambda p0, pts: 1.0 - np.matmul(p0[None, :], pts.T).squeeze(axis=0)
        farthest_idx = np.zeros(self.n_clusters, dtype=np.int)
        farthest_idx[0] = np.random.randint(len(pts))
        distances = calc_distances(pts[farthest_idx[0]], pts)
        for i in range(1, self.n_clusters):
            farthest_idx[i] = np.argmax(distances, axis=0)
            farthest_pts = pts[farthest_idx[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts, pts))
        return farthest_idx

    def fps_euc(self, pts):
        calc_distances = lambda p0, pts: ((p0 - pts) ** 2).sum(axis=1)
        farthest_idx = np.zeros(self.n_clusters, dtype=np.int)
        farthest_idx[0] = np.random.randint(len(pts))
        distances = calc_distances(pts[farthest_idx[0]], pts)
        for i in range(1, self.n_clusters):
            farthest_idx[i] = np.argmax(distances, axis=0)
            farthest_pts = pts[farthest_idx[i]]
            distances = np.minimum(distances, calc_distances(farthest_pts, pts))
        return farthest_idx

    def visualize(self, pts, seg, centers_ids):
        self.cmap = create_ade20k_label_colormap()
        vis = o3d.visualization.Visualizer()
        pcd1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        pcd2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pts))
        pcd1_color = o3d.utility.Vector3dVector(self.cmap[seg])
        pcd2_color = np.ones((len(pts), 3)) * 0.8
        pcd2_color[centers_ids] = np.array([1.0, 0.0, 0.0])
        pcd1.colors = o3d.utility.Vector3dVector(pcd1_color)
        pcd2.colors = o3d.utility.Vector3dVector(pcd2_color)
        vis.create_window()
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2.translate([1.0, 0.0, 0.0]))
        vis.run()
        vis.destroy_window()

    def fit_predict(self, X, verts):
        # init_seeds = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        # seeds_idx = self.fps_embedding(X)
        seeds_idx = self.fps_euc(verts)
        self.centers_emb = X[seeds_idx]
        self.centers_euc = verts[seeds_idx]
        dist_mat = self.calc_dist(X, verts, self.centers_emb, self.centers_euc)
        fit_dist_last = np.min(dist_mat, axis=1).sum()
        for t in range(self.max_iter):
            nnidx_p2s = np.argmin(dist_mat, axis=1)
            # self.visualize(verts, nnidx_p2s, seeds_idx)
            nn_sample2vtx = np.argmin(dist_mat, axis=0)
            for n in range(self.n_clusters):
                associate_ids = np.argwhere(nnidx_p2s == n).squeeze(axis=1)
                if len(associate_ids) == 0:
                    self.centers_emb[n] = X[nn_sample2vtx[n]]
                    self.centers_euc[n] = verts[nn_sample2vtx[n]]
                else:
                    self.centers_emb[n] = np.mean(X[associate_ids], axis=0)
                    self.centers_euc[n] = np.mean(verts[associate_ids], axis=0)
            dist_mat = self.calc_dist(X, verts, self.centers_emb, self.centers_euc)
            fit_dist_this = np.min(dist_mat, axis=1).sum()
            # print(t, fit_dist_last - fit_dist_this, fit_dist_this)
            if np.abs(fit_dist_last - fit_dist_this) < self.tol:
                break
            fit_dist_last = fit_dist_this
        membership = np.argmin(dist_mat, axis=1)
        n_membership = np.bincount(membership)
        self.centers_euc = np.array([self.centers_euc[n] for n in range(len(n_membership)) if n_membership[n] > 8])
        self.centers_emb = np.array([self.centers_emb[n] for n in range(len(n_membership)) if n_membership[n] > 8])
        dist_mat = self.calc_dist(X, verts, self.centers_emb, self.centers_euc)
        return np.argmin(dist_mat, axis=1)