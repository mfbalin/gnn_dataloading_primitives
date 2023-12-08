import itertools
from enum import Enum

import dgl
import dgl.graphbolt as gb
import torch

class Device(Enum):
    GPU = 0
    Pinned = 1


def gen_random_indices(n_rows, num_indices):
    indices = []
    for i in range(50):
        indices.append(torch.randint(0, n_rows, (num_indices,)))
    return indices


def test_index_select_throughput(feature, indices):
    # Warm up
    for _ in range(3):
        for index in indices:
            torch.ops.graphbolt.index_select(feature, index)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in indices:
        torch.ops.graphbolt.index_select(feature, index)
    end.record()
    end.synchronize()
    # Summarize all index sizes
    num_indices = sum([index.numel() for index in indices]) / len(indices)
    average_time = start.elapsed_time(end) / 1000 / len(indices)
    feat_size = feature.shape[1]
    selected_size = num_indices * feat_size * feature.element_size()
    return average_time, selected_size / average_time


available_RAM = 10 * (2**30)  ## 10 GiB
n_rows = [2000000 * factor for factor in [1, 8, 64, 512]]
feat_size = [1, 4, 16, 47, 188, 256, 353, 1024, 1412]
num_indices = [1000, 100000, 1000000]
dtypes = [torch.int8]
feature_devices = [Device.Pinned]
indices_devices = [Device.GPU]
keys = [
    "n_rows",
    "feat_size",
    "dtype",
    "num_indices",
    "feature_device",
    "indices_device",
]

sum_of_runtimes = 0


def _print_result(runtime, throughput):
    print(
        f"Runtime in us: {int(runtime * 1000000)}, Throughput in MiB/s: {int(throughput / (2 ** 20))}"
    )
    print("")
    print("")
    global sum_of_runtimes
    sum_of_runtimes += runtime


def test_random():
    for rows, size, feature_device, dtype in itertools.product(
        n_rows, feat_size, feature_devices, dtypes
    ):
        if (
            rows * size * torch.tensor([], dtype=dtype).element_size()
            >= available_RAM
        ):
            continue
        torch.cuda.empty_cache()
        feature = torch.randint(0, 13, size=[rows, size], dtype=dtype)
        feature = (
            feature.cuda()
            if feature_device == Device.GPU
            else feature.pin_memory()
        )
        for indices_size, indices_device in itertools.product(
            num_indices, indices_devices
        ):
            indices = gen_random_indices(rows, indices_size)
            indices = [
                index.cuda()
                if indices_device == Device.GPU
                else index.pin_memory()
                for index in indices
            ]
            params = (
                rows,
                size,
                dtype,
                indices_size,
                feature_device,
                indices_device,
            )
            print(
                "* params: ",
                ", ".join([f"{k}={v}" for k, v in zip(keys, params)]),
            )
            print("")
            params_dict = {
                "feature": feature,
                "indices": indices,
            }
            runtime, throughput = test_index_select_throughput(**params_dict)
            _print_result(runtime, throughput)


test_random()
print("Total runtimes in us: ", int(sum_of_runtimes * 1000000))