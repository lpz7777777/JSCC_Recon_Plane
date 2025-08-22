import torch
import time

def get_compton_backproj_list(list_origin, delta_r1, delta_r2, e0, ene_resolution, ene_threshold_max, ene_threshold_min, detector, coor, sysmat, device):
    cpnum1 = list_origin[:, 0].int()
    cpnum2 = list_origin[:, 2].int()
    e1 = list_origin[:, 1]
    e2 = list_origin[:, 3]

    # set_energy_resolution
    sigma_1 = e1 * ene_resolution / 2.355 * (e0 / e1) ** 0.5
    sigma_2 = e2 * ene_resolution / 2.355 * (e0 / e2) ** 0.5
    e1 += sigma_1 * torch.randn(e1.shape[0]).to(device)
    e2 += sigma_2 * torch.randn(e2.shape[0]).to(device)

    # set_energy_threshold
    flag_max_1 = e1 < ene_threshold_max
    # flag_max_2 = e2 < ene_threshold_max
    flag_min_1 = e1 > ene_threshold_min
    flag_min_2 = e2 > ene_threshold_min
    flag_sum = (e1 + e2) > 0.60
    # flag_tmp_1 = (cpnum1 % 196) != (cpnum2 % 196)

    flag = flag_max_1 * flag_min_1 * flag_min_2 * flag_sum
    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]

    # det_pos
    pos1 = detector[cpnum1 - 1, :]
    pos2 = detector[cpnum2 - 1, :]
    flag = abs(pos1[:, 2] - pos2[:, 2])>0.1

    cpnum1 = cpnum1[flag]
    cpnum2 = cpnum2[flag]
    e1 = e1[flag]
    e2 = e2[flag]
    pos1 = pos1[flag]
    pos2 = pos2[flag]
    # 01: fov pixels to 1st detector
    # 12: 1st detector to 2nd detector
    vector01 = pos1.unsqueeze(1) - coor.unsqueeze(0)
    vector12 = (pos2 - pos1).unsqueeze(1)
    distance01 = torch.norm(vector01, dim=2)
    distance12 = torch.norm(vector12, dim=2)

    # get_scatter_angle
    ee = 0.511
    theta = torch.acos(1 - ((ee * e1) / ((e0 - e1) * e0)))
    # calculate the angular error contributed by energy
    er = e1 * ene_resolution / 2.355 * (e0 / e1) ** 0.5
    angle_sigma_ene = er * (1 / torch.abs(torch.sin(theta))) * ee / (e0 - e1) ** 2
    klein_nishina = e0 / (e0 - e1) + (e0 - e1) / e0

    # get_angle_sigma
    p = distance01 / distance12
    q = delta_r1 / delta_r2
    angle_sigma_pos = torch.atan(delta_r1 / distance01) * (1 + p ** 2 * (1 + q ** 2) + 2 * p * torch.cos(theta.unsqueeze(-1))) ** 0.5
    angle_sigma = (angle_sigma_pos ** 2 + angle_sigma_ene.unsqueeze(-1) ** 2) ** 0.5

    # get_back_proj
    beta = torch.acos((vector01 * vector12).sum(2) / (distance01 * distance12))
    t = torch.exp(- (beta - theta.unsqueeze(-1)) ** 2 / (2 * angle_sigma ** 2)) / ((2*3.14159) ** 0.5 * angle_sigma) * (klein_nishina.unsqueeze(-1) - torch.sin(beta) ** 2)
    t_compton = t
    t_single = sysmat[cpnum1 - 1, :]

    # get t
    t = t * sysmat[cpnum1 - 1, :]
    flag_nan = torch.isnan(t).sum(dim=1)
    flag_zero = (t.sum(dim=1) == 0)

    t = t[(flag_nan + flag_zero) == 0, :]
    t_compton = t_compton[(flag_nan + flag_zero) == 0, :]
    t_single = t_single[(flag_nan + flag_zero) == 0, :]

    t = (t / t.sum(dim=1, keepdim=True)).cpu()
    t_compton = (t_compton / t_compton.sum(dim=1, keepdim=True)).cpu()
    t_single = (t_single / t_single.sum(dim=1, keepdim=True)).cpu()

    return t, t_compton, t_single