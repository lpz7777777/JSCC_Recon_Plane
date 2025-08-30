import torch
import numpy as np
from process_list_plane import get_coor_plane, get_compton_backproj_list_mp
from recon_mlem_plane import run_recon_mlem
import time
import argparse
import sys
import os
import shutil
import torch.multiprocessing as mp
from torch.multiprocessing import Manager


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


def main():
    global start_time
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()

    # File paths
    data_file_path = "ContrastPhantom_70_440keV_5e9"
    factor_file_path = "100_100_3_3_440keV"

    # System factors
    e0 = 0.440
    ene_resolution_662keV = 0.1
    ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
    ene_threshold_max = 2 * e0 ** 2 / (0.511 + 2 * e0) - 0.001
    ene_threshold_min = 0.05

    # FOV factors
    fov_z = -146.5
    pixel_num_x = 100
    pixel_num_y = 100
    pixel_l_x = 3
    pixel_l_y = 3
    pixel_num = pixel_num_x * pixel_num_y

    # Spatial resolution
    delta_r1 = 1.25
    delta_r2 = 1.25
    alpha = 1

    # Reconstruction factors
    iter_arg = argparse.ArgumentParser().parse_args()
    iter_arg.sc = 200
    iter_arg.jsccd = 100
    iter_arg.jsccsd = 200
    iter_arg.save_iter_step = 10
    iter_arg.osem_subset_num = 8
    iter_arg.t_divide_num = 8
    iter_arg.event_level = 2
    iter_arg.num_workers = 20  # Sub-chunks per GPU, to avoid overload of GPUs

    # Downsampling flags
    flag_ds = 1
    ds = 0.1
    flag_save_t = 0
    flag_save_s = 0

    # Setup logging
    logfile = open("print_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, logfile)

    print("=======================================")
    print("--------Step1: Checking Devices--------")
    if torch.cuda.is_available():
        print(f"CUDA is available, found {args.num_gpus} GPUs")
        if args.num_gpus > 1:
            # if number of GPU > 1, then change t_divide_num to GPU number
            iter_arg.t_divide_num = args.num_gpus
    else:
        print("CUDA is not available, running on CPU")
        args.num_gpus = 0


    print("====================================")
    print("--------Step2: Loading Files--------")
    # Load factors
    sysmat_file_path = "./Factors/" + factor_file_path + "/SysMat"
    detector_file_path = "./Factors/" + factor_file_path + "/Detector.csv"
    sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0, 1)
    detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])

    # Load data
    proj_file_path = "./CntStat/CntStat_" + data_file_path + ".csv"
    list_file_path = "./List/List_" + data_file_path + ".csv"
    proj = torch.from_numpy(
        np.reshape(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32), [1, -1])).transpose(0, 1)
    list_origin = torch.from_numpy(np.genfromtxt(list_file_path, delimiter=",", dtype=np.float32)[:, 0:4])


    print("========================================")
    print("--------Step3: Data Downsampling--------")
    if flag_ds == 1:
        # porj
        print("Downsampling On")
        proj_s_ds = torch.zeros(size=proj.size(), dtype=torch.float32)
        proj_s_index = torch.tensor([i for i in range(proj.size(0)) for _ in range(round(proj[i].item()))])
        indices = torch.randperm(proj_s_index.size(0))
        selected_indices = indices[0:int(torch.round(proj.sum() * ds).item())]
        proj_s_index_ds = proj_s_index[selected_indices]

        for i in range(0, proj_s_ds.size(dim=0)):
            proj_s_ds[i] = (proj_s_index_ds == i).sum()
        proj = proj_s_ds

        # list
        indices = torch.randperm(list_origin.size(0))
        selected_indices = indices[0:int(list_origin.size(0) * ds)]
        list_origin = list_origin[selected_indices, :]
    else:
        print("Downsampling Off")


    print("==================================================")
    print("--------Step4: Processing List (Multi-GPU)--------")
    # Prepare coordinate plane
    coor_plane = get_coor_plane(pixel_num_x, pixel_num_y, pixel_l_x, pixel_l_y, fov_z)

    # Split list data across GPUs
    if args.num_gpus > 1:
        chunks = torch.chunk(list_origin, args.num_gpus, dim=0)
    else:
        chunks = [list_origin]

    # Create manager and shared dictionary
    with Manager() as manager:
        result_dict = manager.dict()
        processes = []

        # Start worker processes
        for rank in range(args.num_gpus):
            p = mp.Process(
                target=get_compton_backproj_list_mp,
                args=(rank, args.num_gpus, sysmat, detector, coor_plane, chunks[rank], delta_r1, delta_r2, e0, ene_resolution,
                    ene_threshold_max, ene_threshold_min, result_dict, iter_arg.num_workers, start_time, flag_save_t)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Collect results from shared dictionary
        t_results = []
        for rank in range(args.num_gpus):
            if rank in result_dict:
                t_part = result_dict[rank]
                t_results.append(t_part)
                print(f"Collected result from rank {rank}")
            else:
                print(f"Warning: No result found for rank {rank}")

        # Combine results from all GPUs
        if t_results:
            t = torch.cat(t_results, dim=0)
            compton_event_count = t.size(0)
            size_t = t.element_size() * t.nelement()
        else:
            print("Error: No results collected from any GPU")
            return

    # Create proj with equal count
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
    # Calculate sensitivity map
    s_map_arg = argparse.ArgumentParser().parse_args()
    s_map_arg.s = torch.sum(sysmat, dim=0, keepdim=True).transpose(0, 1)
    s_map_arg.d = s_map_arg.s * compton_event_count / single_event_count

    # Save sensitivity if needed
    if flag_save_s == 1:
        with open("sensitivity_s", "w") as file:
            s_map_arg.s.cpu().numpy().astype('float32').tofile(file)

    torch.cuda.empty_cache()

    # Prepare save path
    save_path = f"./Figure/{data_file_path}_{ds}_Delta{delta_r1}_Alpha{alpha}_ER{ene_resolution_662keV}_OSEM{iter_arg.osem_subset_num}_ITER{iter_arg.jsccsd}_SDU{single_event_count}_DDU{compton_event_count}/"
    os.makedirs(save_path, exist_ok=True)

    # Run reconstruction on GPU 0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_recon_mlem(sysmat, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path, args.num_gpus)

    print(f"\nTotal time used: {time.time() - start_time:.2f}s")

    # Cleanup
    logfile.close()
    sys.stdout = sys.__stdout__
    shutil.move("print_log.txt", os.path.join(save_path, "print_log.txt"))


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
