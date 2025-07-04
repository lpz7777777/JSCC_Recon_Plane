import torch
import numpy as np
from process_list_plane import get_compton_backproj_list
from recon_mlem_plane import run_recon_mlem
import time
from scipy.io import loadmat
import argparse
import pickle
import sys
import os
import shutil

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


def get_coor_plane(pixel_num_x, pixel_num_y, pixel_l_x, pixel_l_y, fov_z):
    # get the coordinate of fov
    fov_coor = torch.ones([pixel_num_x, pixel_num_y, 3])
    min_x = -(pixel_num_x / 2 - 0.5) * pixel_l_x
    max_x = (pixel_num_x / 2 - 0.5) * pixel_l_x
    min_y = -(pixel_num_y / 2 - 0.5) * pixel_l_y
    max_y = (pixel_num_y / 2 - 0.5) * pixel_l_y
    fov_coor[:, :, 0] *= torch.linspace(min_x, max_x, pixel_num_x).reshape([1, -1])
    fov_coor[:, :, 1] *= torch.linspace(min_y, max_y, pixel_num_y).reshape([-1, 1])
    fov_coor[:, :, 2] = fov_z
    fov_coor = fov_coor.reshape(-1, 3)
    return fov_coor


if __name__ == '__main__':
    with torch.no_grad():
        # file path
        data_file_path = "ContrastPhantom_140_1e9_1"
        factor_file_path = "100_100_5_5_662keV"

        # set system factors
        e0 = 0.662  # energy of incident photons
        ene_resolution_662keV = 0.1  # energy resolution at 662keV
        ene_resolution = ene_resolution_662keV * (0.662 / e0) ** 0.5
        ene_threshold_max = 0.477
        ene_threshold_min = 0.050

        # fov factor
        fov_z = -146.5
        pixel_num_x = 100
        pixel_num_y = 100
        pixel_l_x = 5  # unit: mm
        pixel_l_y = 5
        pixel_num = pixel_num_x * pixel_num_y

        # intrinsic spatial resolution of scintillators
        delta_r1 = 1.25
        delta_r2 = 1.25
        alpha = 1

        # divide list-mode data into subsets to prevent GPU overload
        num_workers = 20

        # reconstruction factors
        iter_arg = argparse.ArgumentParser().parse_args()
        iter_arg.sc = 2000  # CntStat only
        iter_arg.jsccd = 1000  # List only
        iter_arg.jsccsd = 2000  # CntStat+List
        iter_arg.save_iter_step = 10
        iter_arg.osem_subset_num = 8
        iter_arg.t_divide_num = 5       # prevent memory explosions during iterations
        iter_arg.event_level = 2

        # down sampling ratio of events
        flag_ds = 0
        ds = 1

        # whether to store t
        flag_save_t = 0
        flag_save_s = 0

        # --------Step1: Checking Devices--------
        time_start = time.time()
        # start getting outputs
        logfile = open("print_log.txt", "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, logfile)

        print("")
        print("--------Step1: Checking Devices--------")
        print("Checking Devices starts")
        # judge if CUDA is available and set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available, running on GPU")
        else:
            device = torch.device("cpu")
            print("CUDA is not available, running on CPU")

        print("Checking Devices ends, time used:", time.time() - time_start, "s")

        # --------Step2: Loading Files--------
        print("--------Step2: Loading Files--------")
        print("Loading Files starts")

        # Factors
        sysmat_file_path = "./Factors/" + factor_file_path + "/SysMat"
        detector_file_path = "./Factors/" + factor_file_path + "/Detector.csv"
        sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0,1)
        detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])

        # Data
        proj_file_path = "./CntStat/CntStat_" + data_file_path + ".csv"
        list_file_path = "./List/List_" + data_file_path + ".csv"
        proj = torch.from_numpy(np.reshape(np.genfromtxt(proj_file_path, delimiter=",", dtype=np.float32), [1, -1])).transpose(0, 1)
        list_origin = torch.from_numpy(np.genfromtxt(list_file_path, delimiter=",", dtype=np.float32)[:, 0:4])

        print("Loading Files ends, time used:", time.time() - time_start, "s")

        # --------Step3: Data Downsampling--------
        print("--------Step3: Data Downsampling--------")
        print("Data Downsampling starts")

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

        print("Data Downsampling ends, time used:", time.time() - time_start, "s")

        # --------Step4: Processing List--------
        print("--------Step4: Processing List--------")
        print("Processing List starts")

        # get the coordinate and to cuda
        sysmat = sysmat.to(device)
        detector = detector.to(device)
        coor_plane = get_coor_plane(pixel_num_x, pixel_num_y, pixel_l_x, pixel_l_y, fov_z).to(device)

        if flag_save_t == 1:
            t = []
            t_compton = []
            t_single = []
            list_origin_chunks = torch.chunk(list_origin, num_workers, dim=0)

            for list_origin_tmp_chunk in list_origin_chunks:
                t_chunk, t_compton_chunk, t_single_chunk = get_compton_backproj_list(list_origin_tmp_chunk.to(device), delta_r1, delta_r2, e0, ene_resolution,
                                                            ene_threshold_max, ene_threshold_min, detector, coor_plane, sysmat, device)
                t.append(t_chunk)
                t_compton.append(t_compton_chunk)
                t_single.append(t_single_chunk)
                torch.cuda.empty_cache()
                print("Chunk Num", str(len(t)), "ends, time used:", time.time() - time_start, "s")

            t = torch.cat(t, dim=0)
            t_compton = torch.cat(t_compton, dim=0)
            t_single = torch.cat(t_single, dim=0)

            compton_event_count = t.size(0)
            size_t = t.element_size() * t.nelement()

            # create a proj that has an equal count
            proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
            proj_s_index = torch.tensor([i for i in range(proj.size(0)) for _ in range(round(proj[i].item()))])
            indices = torch.randperm(proj_s_index.size(0))
            selected_indices = indices[0:compton_event_count]
            proj_d_index = proj_s_index[selected_indices]

            for i in range(0, proj_d.size(dim=0)):
                proj_d[i] = (proj_d_index == i).sum()

            # save t
            t_save_path = "./Backproj/" + data_file_path + "/JSCC"
            t_compton_save_path = "./Backproj/" + data_file_path + "/ComptonCone"
            t_single_save_path = "./Backproj/" + data_file_path + "/SysMat"
            if not os.path.exists(t_save_path):
                os.makedirs(t_save_path)
                os.makedirs(t_compton_save_path)
                os.makedirs(t_single_save_path)

            print("Saving t")
            # JSCC
            with open(t_save_path, "w") as file:
                t.transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

            # Compton Cone
            with open(t_compton_save_path, "w") as file:
                t_compton.transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

            # SysMat
            with open(t_single_save_path, "w") as file:
                t_single.transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

        else:
            t = []
            list_origin_chunks = torch.chunk(list_origin, num_workers, dim=0)

            for list_origin_tmp_chunk in list_origin_chunks:
                t_chunk, _, _ = get_compton_backproj_list(list_origin_tmp_chunk.to(device), delta_r1, delta_r2, e0, ene_resolution,
                                                            ene_threshold_max, ene_threshold_min, detector, coor_plane, sysmat, device)
                t.append(t_chunk)
                torch.cuda.empty_cache()
                print("Chunk Num", str(len(t)), "ends, time used:", time.time() - time_start, "s")

            t = torch.cat(t, dim=0)

            compton_event_count = t.size(0)
            size_t = t.element_size() * t.nelement()

            # create a proj that has an equal count
            proj_d = torch.zeros(size=proj.size(), dtype=torch.float32)
            proj_s_index = torch.tensor([i for i in range(proj.size(0)) for _ in range(round(proj[i].item()))])
            indices = torch.randperm(proj_s_index.size(0))
            selected_indices = indices[0:compton_event_count]
            proj_d_index = proj_s_index[selected_indices]

            for i in range(0, proj_d.size(dim=0)):
                proj_d[i] = (proj_d_index == i).sum()

        sysmat = sysmat.cpu()
        del detector, coor_plane

        single_event_count = round(proj.sum().item())
        print("single events = ", single_event_count, ", Compton events = ", compton_event_count)
        print("The size of t is ", size_t / (1024 **3), " GB")
        print("Processing List ends, time used:", time.time() - time_start, "s")

        # --------Step5: Image Reconstruction--------
        print("--------Step5: Image Reconstruction--------")
        print("Image Reconstruction List starts")

        # calculate sensitivity map
        s_map_arg = argparse.ArgumentParser().parse_args()
        s_map_arg.s = torch.sum(sysmat, dim=0, keepdim=True).transpose(0, 1)
        s_map_arg.d = s_map_arg.s * compton_event_count / single_event_count

        # save sensitivity
        if flag_save_s == 1:
            with open("sensitivity_s", "w") as file:
                s_map_arg.s.cpu().numpy().astype('float32').tofile(file)
        torch.cuda.empty_cache()

        save_path = "./Figure/" + data_file_path + "_" + str(ds) + "_Delta" + str(delta_r1) + "_ER" + str(ene_resolution_662keV) + "_OSEM" + str(iter_arg.osem_subset_num) + "_ITER" + str(iter_arg.jsccsd) + "_SDU" + str(single_event_count) + "_DDU" + str(compton_event_count) + "/"

        run_recon_mlem(sysmat, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path, device)

        print("Image Reconstruction ends, time used:", time.time() - time_start, "s")
        print("Total time used:", time.time() - time_start)

        # get all outputs
        logfile.close()
        sys.stdout = sys.__stdout__
        shutil.move("print_log.txt", save_path + "print_log.txt")
