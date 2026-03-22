import os
import json
import torch
from argparse import ArgumentParser

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, HpatchesHomogBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name, data_root="data/megadepth", scene_names=None, output=None,
                  dump_dir=None, max_pairs=None):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark(data_root, scene_names=scene_names)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name,
                                                    dump_dir=dump_dir, max_pairs=max_pairs)
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(mega1500_results, f, indent=2)
    else:
        json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))
    return mega1500_results

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"results/hpatches_{name}.json", "w"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "16"

    from romatch import roma_outdoor

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to roma_outdoor.pth checkpoint (raw state dict)")
    parser.add_argument("--data_root", default="data/megadepth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_dir", default=None,
                        help="Directory to dump preprocessed tensors and sparse matches for wrapper verification")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Limit number of pairs processed (for wrapper verification)")
    args, _ = parser.parse_known_args()

    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('highest')

    device = "cuda"
    weights = torch.load(args.checkpoint, map_location=device)
    model = roma_outdoor(device=device, weights=weights, coarse_res=672, upsample_res=1344,
                         use_custom_corr=False)
    experiment_name = "roma_outdoor"
    test_mega1500(model, experiment_name, data_root=args.data_root, output=args.output,
                  dump_dir=args.dump_dir, max_pairs=args.max_pairs)
    #test_hpatches(model, experiment_name)
    #test_mega1500_poselib(model, experiment_name)

