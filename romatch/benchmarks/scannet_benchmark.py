import os
import os.path as osp
import json
import numpy as np
import torch
from romatch.utils import *
from romatch.utils.utils import get_tuple_transform_ops
from PIL import Image
from tqdm import tqdm


class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root

    def benchmark(self, model, model_name=None, seed=0, dump_dir=None, max_pairs=None, output=None):
        model.train(False)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)

        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            if max_pairs is not None:
                pair_inds = pair_inds[:max_pairs]

            device = next(model.parameters()).device
            hs, ws = model.h_resized, model.w_resized
            test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True, clahe=False)

            for dump_idx, pairind in enumerate(tqdm(pair_inds, smoothing=0.9)):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im_A_path = osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                im_A = Image.open(im_A_path)
                im_B_path = osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                im_B = Image.open(im_B_path)
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            osp.join(
                                self.data_root,
                                "scans_test",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )
                w1, h1 = im_A.size
                w2, h2 = im_B.size
                K1 = K.copy()
                K2 = K.copy()

                im_A_t, im_B_t = test_transform((im_A.convert("RGB"), im_B.convert("RGB")))
                dense_matches, dense_certainty = model.match(
                    im_A_t[None].to(device), im_B_t[None].to(device)
                )
                sparse_matches, sparse_certainty = model.sample(
                    dense_matches, dense_certainty, 5000
                )
                sparse_matches_np = sparse_matches.cpu().numpy()

                if dump_dir is not None:
                    torch.save({
                        'image0': im_A_t.cpu(),
                        'image1': im_B_t.cpu(),
                        'sparse_matches': sparse_matches.cpu(),
                        'im0_path': im_A_path,
                        'im1_path': im_B_path,
                    }, osp.join(dump_dir, f'{dump_idx:05d}.pt'))

                scale1 = 480 / min(w1, h1)
                scale2 = 480 / min(w2, h2)
                w1, h1 = scale1 * w1, scale1 * h1
                w2, h2 = scale2 * w2, scale2 * h2
                K1 = K1 * scale1
                K2 = K2 * scale2

                offset = 0.5
                kpts1 = sparse_matches_np[:, :2]
                kpts1 = (
                    np.stack(
                        (
                            w1 * (kpts1[:, 0] + 1) / 2 - offset,
                            h1 * (kpts1[:, 1] + 1) / 2 - offset,
                        ),
                        axis=-1,
                    )
                )
                kpts2 = sparse_matches_np[:, 2:]
                kpts2 = (
                    np.stack(
                        (
                            w2 * (kpts2[:, 0] + 1) / 2 - offset,
                            h2 * (kpts2[:, 1] + 1) / 2 - offset,
                        ),
                        axis=-1,
                    )
                )
                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]
                    try:
                        norm_threshold = 0.5 / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        T1_to_2_est = np.concatenate((R_est, t_est), axis=-1)
                        e_t, e_R = compute_pose_error(T1_to_2_est, R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        print(repr(e))
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
                tot_e_t.append(e_t)
                tot_e_R.append(e_R)
                tot_e_pose.append(e_pose)

            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            results = {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }

            if output is not None:
                os.makedirs(osp.dirname(osp.abspath(output)), exist_ok=True)
                with open(output, 'w') as f:
                    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

            return results
