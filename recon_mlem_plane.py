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
    return torch.nan_to_num(t_List_tmp.transpose(0, 1) / (torch.matmul(t_List_tmp, img_tmp).sum(1)), nan=0, posinf=0, neginf=0).sum(1, keepdim=True).to("cuda:0")

def get_weight_compton_mp(q, t_List_tmp, img_tmp):
    # for multiprocessing
    q.put(get_weight_compton(t_List_tmp, img_tmp))


def mlem_bin_mode(sysmat_list, proj_list, img, s_map, osem_subset_num):
    # mlem algorithm
    for i in range(osem_subset_num):
        weight = get_weight_single(sysmat_list[i], proj_list[i], img)
        img = img * weight / s_map

    return img


def mlem_list_mode(t_list, img, s_map, osem_subset_num, t_divide_num):
    # list mode mlem algorithm (No MultiProcess)
    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(t_divide_num):
            weight_tmp = get_weight_compton(t_list[i][j], img)
            weight = weight + weight_tmp
        img = img * weight / s_map

    return img


def mlem_list_mode_mp(t_list, img, s_map, osem_subset_num, t_divide_num):
    # list mode mlem algorithm (With MultiProcess)
    # prepare for mp
    q = mp.Queue()
    processes = []

    for i in range(osem_subset_num):
        weight = 0 * img
        for j in range(t_divide_num):
            p = mp.Process(target=get_weight_compton_mp, args=(q, t_list[i][j], img.to(f"cuda:{j}", non_blocking=True)))
            p.start()
            processes.append(p)

        for _ in range(t_divide_num):
            weight_tmp = q.get()
            weight = weight + weight_tmp

        for p in processes:
            p.join()

        img = img * weight / s_map

    return img


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


def mlem_joint_mode_mp(sysmat_list, proj_list, t_list, img, s_map, osem_subset_num, t_divide_num, alpha):
    # joint mode mlem algorithm (With MultiProcess)
    # prepare for mp
    q = mp.Queue()
    processes = []

    for i in range(osem_subset_num):
        weight_compton = 0 * img
        weight_single = get_weight_single(sysmat_list[i], proj_list[i], img)

        for j in range(t_divide_num):
            p = mp.Process(target=get_weight_compton_mp, args=(q, t_list[i][j], img.to(f"cuda:{j}", non_blocking=True)))
            p.start()
            processes.append(p)

        for _ in range(t_divide_num):
            weight_compton_tmp = q.get()
            weight_compton = weight_compton + weight_compton_tmp

        for p in processes:
            p.join()

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
        t_list = list(torch.chunk(t, iter_arg.osem_subset_num, dim=0))
        for i in range(0, iter_arg.osem_subset_num):
            t_list[i] = list(torch.chunk(t_list[i], iter_arg.t_divide_num, dim=0))
            for j in range(0, iter_arg.t_divide_num):
                t_list[i][j] = t_list[i][j].to(f"cuda:{j}", non_blocking=True)
    
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
        # jscc-d
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

        # jscc-sd
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
        # jscc-d
        print("JSCC-D MLEM starts (With Multiprocessing)")
        id_save = 0
        for id_iter_jsccd in range(iter_arg.jsccd):
            img_jsccd = mlem_list_mode_mp(t_list, img_jsccd, s_map_arg.d, iter_arg.osem_subset_num, iter_arg.t_divide_num)
            if (id_iter_jsccd + 1) % iter_arg.save_iter_step == 0:
                img_jsccd_iter[id_save, :] = torch.squeeze(img_jsccd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-D MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

        # jscc-sd
        print("JSCC-SD MLEM starts (With Multiprocessing)")
        id_save = 0
        for id_iter_jsccsd in range(iter_arg.jsccsd):
            img_jsccsd = mlem_joint_mode_mp(sysmat_list, proj_list, t_list, img_jsccsd, alpha * s_map_arg.s + (2 - alpha) * s_map_arg.d, iter_arg.osem_subset_num, iter_arg.t_divide_num, alpha)
            if (id_iter_jsccsd + 1) % iter_arg.save_iter_step == 0:
                img_jsccsd_iter[id_save, :] = torch.squeeze(img_jsccsd).cpu()
                id_save += 1
                print("Iteration ", str(id_iter_jsccsd + 1), " ends, time used:", time.time() - time_start, "s")

        print("JSCC-SD MLEM ends, time used:", time.time() - time_start)
        torch.cuda.empty_cache()

    # save images as binary file to 'Figure'
    save_img(img_sc, img_scd, img_jsccd, img_jsccsd, img_sc_iter, img_scd_iter, img_jsccd_iter, img_jsccsd_iter, iter_arg, save_path)