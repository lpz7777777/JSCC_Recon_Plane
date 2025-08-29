import torch
import numpy as np
import time
import os
import torch.multiprocessing as mp

def get_weight_single(sysmat_list_tmp, proj_list_tmp, img_tmp):
    # get the weight of single events
    return torch.matmul(sysmat_list_tmp.transpose(0, 1), proj_list_tmp / torch.matmul(sysmat_list_tmp, img_tmp))

def get_weight_compton(t_List_tmp, img_tmp):
    # get the weight of Compton events
    return torch.nan_to_num(t_List_tmp.transpose(0, 1) / (torch.matmul(t_List_tmp, img_tmp).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True)


def mlem_bin_mode(sysmat_list, proj_list, img, s_map, osem_subset_num):
    # mlem algorithm
    for i in range(osem_subset_num):
        weight = get_weight_single(sysmat_list[i], proj_list[i], img)
        img = img * weight / s_map

    return img


def mlem_list_mode(t_list, img, s_map, osem_subset_num, t_divide_num):
    # list mode mlem algorithm (No MultiProcess)
    for i in range(osem_subset_num):
        weight_compton = 0 * img
        for j in range(t_divide_num):
            weight_tmp = get_weight_compton(t_list[i][j], img)
            weight_compton = weight_compton + weight_tmp
        img = img * weight_compton / s_map

    return img


def mlem_list_mode_mp(t_list, iter_num, osem_subset_num, rank, img_queue, weight_queue):
    # list mode mlem algorithm (With MultiProcess)
    print(f"List Mode MLEM Rank{rank} Starts")
    for id_iter in range(iter_num):
        for i in range(osem_subset_num):
            img = img_queue.get()
            weight = get_weight_compton(t_list[i], img.to(f"cuda:{rank}")).to("cuda:0")
            weight_queue.put(weight)


def mlem_joint_mode(sysmat_list, proj_list, t_list, img, s_map, osem_subset_num, t_divide_num, alpha):
    # joint mode mlem algorithm (No MultiProcess)
    for i in range(osem_subset_num):
        weight_compton = 0 * img
        weight_single = get_weight_single(sysmat_list[i], proj_list[i], img)

        for j in range(t_divide_num):
            weight_compton_tmp = get_weight_compton(t_list[i][j], img)
            weight_compton = weight_compton + weight_compton_tmp
        weight = (2 - alpha) * weight_compton + alpha * weight_single
        img = img * weight / s_map

    return img


def save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path):
    with open(save_path + "Image_SC", "wb") as file:
        img_sc.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD", "wb") as file:
        img_scd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD", "wb") as file:
        img_jsccd.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD", "wb") as file:
        img_jsccsd.cpu().numpy().astype('float32').tofile(file)

    with open(save_path + "Image_SC_Iter_%d_%d" % (iter_arg.sc, iter_arg.sc / iter_arg.save_iter_step), "wb") as file:
        img_sc_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_SCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_scd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCD_Iter_%d_%d" % (iter_arg.jsccd, iter_arg.jsccd / iter_arg.save_iter_step), "wb") as file:
        img_jsccd_iter.cpu().numpy().astype('float32').tofile(file)
    with open(save_path + "Image_JSCCSD_Iter_%d_%d" % (iter_arg.jsccsd, iter_arg.jsccsd / iter_arg.save_iter_step), "wb") as file:
        img_jsccsd_iter.cpu().numpy().astype('float32').tofile(file)

    file.close()


def run_recon_mlem(sysmat, proj, proj_d, t, iter_arg, s_map_arg, alpha, save_path, num_gpus):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pixel_num = sysmat.size(1)

    # run mlem on gpu
    # initial image
    img_sc = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_scd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)
    img_jsccsd = torch.ones([pixel_num, 1], dtype=torch.float32).to("cuda:0", non_blocking=True)

    img_sc_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_scd_iter = torch.ones([round(iter_arg.sc / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccd_iter = torch.ones([round(iter_arg.jsccd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)
    img_jsccsd_iter = torch.ones([round(iter_arg.jsccsd / iter_arg.save_iter_step), pixel_num], dtype=torch.float32)


    # divide datas into subsets and make data to gpu or pin memory
    if num_gpus == 1:
        t_list = list(torch.chunk(t.to("cuda:0", non_blocking=True), iter_arg.osem_subset_num, dim=0))
        for i in range(0, iter_arg.osem_subset_num):
            t_list[i] = list(torch.chunk(t_list[i], iter_arg.t_divide_num, dim=0))

    else:
        t_list = list(torch.chunk(t, iter_arg.t_divide_num, dim=0))
        for i in range(0, iter_arg.t_divide_num):
            t_list[i] = list(torch.chunk(t_list[i].to(f"cuda:{i}"), iter_arg.osem_subset_num, dim=0))
    
    del t

    cpnum_list = torch.arange(0, proj.size(dim=0))
    random_id = torch.randperm(proj.size(dim=0))
    cpnum_list = cpnum_list[random_id]
    cpnum_list = list(torch.chunk(cpnum_list, iter_arg.osem_subset_num, dim=0))

    sysmat_list = []
    proj_list = []
    proj_d_list = []
    for i in range(0, iter_arg.osem_subset_num):
        sysmat_list.append(sysmat[cpnum_list[i], :].to("cuda:0", non_blocking=True))
        proj_list.append(proj[cpnum_list[i], :].to("cuda:0", non_blocking=True))
        proj_d_list.append(proj_d[cpnum_list[i], :].to("cuda:0", non_blocking=True))

    s_map_arg.s = s_map_arg.s.to("cuda:0", non_blocking=True)
    s_map_arg.d = s_map_arg.d.to("cuda:0", non_blocking=True)

    # do iteration
    time_start = time.time()

    # self-collimation
    print("Self-Collimation MLEM starts")
    id_save = 0
    for id_iter_sc in range(iter_arg.sc):
        img_sc = mlem_bin_mode(sysmat_list, proj_list, img_sc, s_map_arg.s, iter_arg.osem_subset_num)
        if (id_iter_sc + 1) % iter_arg.save_iter_step == 0:
            img_sc_iter[id_save, :] = torch.squeeze(img_sc).cpu()
            id_save += 1
            print("Iteration ", str(id_iter_sc + 1), " ends, time used:", time.time() - time_start, "s")

    print("Self-Collimation MLEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    # sc-d
    print("SC-D MLEM starts")
    id_save = 0
    for id_iter_scd in range(iter_arg.jsccd):
        img_scd = mlem_bin_mode(sysmat_list, proj_d_list, img_scd, s_map_arg.d, iter_arg.osem_subset_num)
        if (id_iter_scd + 1) % iter_arg.save_iter_step == 0:
            img_scd_iter[id_save, :] = torch.squeeze(img_scd).cpu()
            id_save += 1
            print("Iteration ", str(id_iter_scd + 1), " ends, time used:", time.time() - time_start, "s")

    print("SC-D MLEM ends, time used:", time.time() - time_start)
    torch.cuda.empty_cache()

    if num_gpus == 1:
        # ========== jscc-d ==========
        print("JSCC-D MLEM starts")
        id_save = 0
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = mlem_list_mode(t_list, img_jsccd, s_map_arg.d, iter_arg.osem_subset_num, iter_arg.t_divide_num)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-D MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

        # ========== jscc-sd ==========
        print("JSCC-SD MLEM starts")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = mlem_joint_mode(sysmat_list, proj_list, t_list, img_jsccsd, alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d, iter_arg.osem_subset_num, iter_arg.t_divide_num, alpha)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-SD MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

    else:
        # prepare queue for mp
        img_queue = mp.Queue()
        weight_queue = mp.Queue()

        # ========== jscc-d ==========
        print("JSCC-D MLEM starts (With Multiprocessing)")
        id_save = 0
        processes = []

        for j in range(iter_arg.t_divide_num):
            p = mp.Process(target=mlem_list_mode_mp, args=(t_list[j], iter_arg.jsccd, iter_arg.osem_subset_num, j, img_queue, weight_queue))
            p.start()
            processes.append(p)

        for id_iter_jsccd in range(iter_arg.jsccd):
            for i in range(iter_arg.osem_subset_num):
                weight_compton = 0 * img_jsccd
                for j in range(iter_arg.t_divide_num):
                    img_queue.put(img_jsccd)
                for j in range(iter_arg.t_divide_num):
                    weight_compton_tmp = weight_queue.get()
                    weight_compton = weight_compton + weight_compton_tmp

                img_jsccd = img_jsccd * weight_compton / s_map_arg.d

            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        for p in processes:
            p.join()
        processes.clear()

        print("JSCC-D MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

        # ========== jscc-sd ==========
        print("JSCC-SD MLEM starts (With Multiprocessing)")
        id_save = 0
        processes = []
        s_map_arg.j = alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d

        for j in range(iter_arg.t_divide_num):
            p = mp.Process(target=mlem_list_mode_mp, args=(t_list[j], iter_arg.jsccsd, iter_arg.osem_subset_num, j, img_queue, weight_queue))
            p.start()
            processes.append(p)

        for id_iter_jsccsd in range(iter_arg.jsccsd):
            for i in range(iter_arg.osem_subset_num):
                weight_compton = 0 * img_jsccsd
                for j in range(iter_arg.t_divide_num):
                    img_queue.put(img_jsccsd)
                for j in range(iter_arg.t_divide_num):
                    weight_compton_tmp = weight_queue.get()
                    weight_compton = weight_compton + weight_compton_tmp

                weight_single = get_weight_single(sysmat_list[i], proj_list[i], img_jsccsd)
                weight = alpha * weight_single + (2 - alpha) * weight_compton

                img_jsccsd = img_jsccsd * weight / s_map_arg.j

            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        for p in processes:
            p.join()
        processes.clear()

        print("JSCC-SD MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

    # save images as binary file to 'Figure'
    save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)