import torch
import numpy as np
from process_list_plane import get_coor_plane, get_compton_backproj_list_dist
from recon_mlem_plane import run_recon_mlem
import time
import argparse
import sys
import os
import shutil
import torch.distributed as dist
import socket


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node = socket.gethostname()

    print(f"[DEBUG] Host={node}, GlobalRank={rank}, LocalRank={local_rank}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"current_device={torch.cuda.current_device()}")

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def main():
    global start_time
    start_time = time.time()

    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()

    data_file_path = "ContrastPhantom_70_440keV_5e9"
    factor_file_path = "100_100_3_3_440keV"

    e0 = 0.440
    ene_resolution_662keV = 0.1
    ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
    ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
    ene_threshold_min = 0.05

    fov_z = -146.5
    pixel_num_x = 100
    pixel_num_y = 100
    pixel_l_x = 3
    pixel_l_y = 3
    pixel_num = pixel_num_x * pixel_num_y

    delta_r1 = 1.25
    delta_r2 = 1.25
    alpha = 1

    iter_arg = argparse.ArgumentParser().parse_args()
    iter_arg.sc = 2000
    iter_arg.jsccd = 1000
    iter_arg.jsccsd = 2000
    iter_arg.save_iter_step = 10
    iter_arg.osem_subset_num = 8
    iter_arg.t_divide_num = 5
    iter_arg.event_level = 2

    flag_ds = 1
    ds = 0.5
    flag_save_t = 0
    flag_save_s = 0

    if is_main_process:
        logfile = open("print_log.txt", "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)

    print(f"Rank {rank}/{world_size} (Local rank: {local_rank})")
    print("=======================================")
    print("--------Step1: Checking Devices--------")
    if torch.cuda.is_available():
        print(f"CUDA is available, found {args.num_gpus} GPUs")
    else:
        print("CUDA is not available, running on CPU")
        args.num_gpus = 0

    print("====================================")
    print("--------Step2: Loading Files--------")
    sysmat_file_path = "./Factors/" + factor_file_path + "/SysMat"
    detector_file_path = "./Factors/" + factor_file_path + "/Detector.csv"
    sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0, 1)
    detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])

    proj_file_path = "./CntStat/CntStat_" + data_file_path + ".csv"
    list_file_path = "./List/List_" + data_file_path + ".csv"
    proj = torch.from_numpy(
        np.reshape(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32), [1, -1])).transpose(0, 1)
    list_origin = torch.from_numpy(np.genfromtxt(list_file_path, delimiter=",", dtype=np.float32)[:, 0:4])

    print("========================================")
    print("--------Step3: Data Downsampling--------")
    if flag_ds == 1:
        print("Downsampling On")
        proj_s_ds = torch.zeros(size=proj.size(), dtype=torch.float32)
        proj_s_index = torch.tensor([i for i in range(proj.size(0)) for _ in range(round(proj[i].item()))])
        indices = torch.randperm(proj_s_index.size(0))
        selected_indices = indices[0:int(torch.round(proj.sum() * ds).item())]
        proj_s_index_ds = proj_s_index[selected_indices]
        for i in range(0, proj_s_ds.size(dim=0)):
            proj_s_ds[i] = (proj_s_index_ds == i).sum()
        proj = proj_s_ds

        indices = torch.randperm(list_origin.size(0))
        selected_indices = indices[0:int(list_origin.size(0) * ds)]
        list_origin = list_origin[selected_indices, :]
    else:
        print("Downsampling Off")

    print("==================================================")
    print("--------Step4: Processing List (Distributed)--------")
    coor_plane = get_coor_plane(pixel_num_x, pixel_num_y, pixel_l_x, pixel_l_y, fov_z)

    total_events = list_origin.size(0)
    events_per_rank = total_events // world_size
    remainder = total_events % world_size
    start_idx = rank * events_per_rank + min(rank, remainder)
    end_idx = start_idx + events_per_rank + (1 if rank < remainder else 0)
    local_list = list_origin[start_idx:end_idx]

    print(f"Rank {rank} processing {local_list.size(0)} events (total: {total_events})")

    t_part = get_compton_backproj_list_dist(
        local_rank, world_size, sysmat, detector, coor_plane, local_list,
        delta_r1, delta_r2, e0, ene_resolution, ene_threshold_max,
        ene_threshold_min, None, 20, start_time, flag_save_t
    )

    # ✅ 用 all_gather_object 收集结果，避免 padding
    local_array = t_part.numpy() if t_part is not None else None
    gathered_list = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_list, local_array)

    if is_main_process:
        t_results = [torch.from_numpy(arr) for arr in gathered_list if arr is not None]
        if t_results:
            t = torch.cat(t_results, dim=0)
            compton_event_count = t.size(0)
            size_t = t.element_size() * t.nelement()
        else:
            print("Error: No results collected from any rank")
            cleanup_distributed()
            return
    else:
        t = None
        compton_event_count = 0

    dist.barrier()

    if is_main_process:
        proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
        proj_s_index = torch.tensor([i for i in range(proj.size(0)) for _ in range(round(proj[i].item()))])
        indices = torch.randperm(proj_s_index.size(0))
        selected_indices = indices[0:compton_event_count]
        proj_d_index = proj_s_index[selected_indices]
        for i in range(proj_d.size(dim=0)):
            proj_d[i] = (proj_d_index == i).sum()

        single_event_count = round(proj.sum().item())
        print(f"Single events = {single_event_count}, Compton events = {compton_event_count}")
        print(f"The size of t is {size_t / (1024 ** 3):.2f} GB")

        print("===========================================")
        print("--------Step5: Image Reconstruction--------")
        s_map_arg = argparse.ArgumentParser().parse_args()
        s_map_arg.s = torch.sum(sysmat, dim=0, keepdim=True).transpose(0, 1)
        s_map_arg.d = s_map_arg.s * compton_event_count / single_event_count

        if flag_save_s == 1:
            with open("sensitivity_s", "w") as file:
                s_map_arg.s.cpu().numpy().astype('float32').tofile(file)

        torch.cuda.empty_cache()

        save_path = f"./Figure/{data_file_path}_{ds}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_OSEM{iter_arg.osem_subset_num}_ITER{iter_arg.jsccsd}_SDU{single_event_count}_DDU{compton_event_count}/"
        os.makedirs(save_path, exist_ok=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_recon_mlem(sysmat, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path, device)

        print(f"\nTotal time used: {time.time() - start_time:.2f}s")

        logfile.close()
        sys.stdout = sys.__stdout__
        shutil.move("print_log.txt", os.path.join(save_path, "print_log.txt"))

    cleanup_distributed()


if __name__ == '__main__':
    main()
