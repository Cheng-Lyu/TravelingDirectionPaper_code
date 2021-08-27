import plotting_help_py3 as ph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

pi = np.pi

# Help functions
def ave_weighted_angles(ws, angles):
    n = len(ws)
    z = 0 + 0j
    for i in range(n):
        a = np.radians(angles[i])
        w = ws[i]
        x = w*np.cos(a)
        y = w*np.sin(a)
        z_ = complex(x, y)
        z = z + z_
    angle = np.angle(z, deg=True)
    return angle

def circsubtract(x, y, low=-180, high=180):
    if len(x) != len(y):
        print('two input lists are of different length ! !')
        return 0
    z = np.zeros(len(x))
    z = np.array(x) - np.array(y)
    for index in range(len(z)):
        if z[index] < low:
            z[index] += 360
        elif z[index] >= high:
            z[index] -= 360
    return np.array(z)

def print_connections_btw_two_celltype(client, celltypes=['FB7B', 'FB6A']):
    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
         WHERE n0.type = '%s' AND n1.type = '%s'
         RETURN n0.bodyId, n0.instance, c.weight AS w, n1.bodyId, n1.instance
         ORDER BY w ascending
         """ % (celltypes[0], celltypes[1])
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    print(data_orig)

def get_bodyId_of_one_celltype(client, celltype='FB4L'):
    q = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId
         """ % (celltype)
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    neuron_ids = []
    for row in data_orig:
        neuron_ids.append(row[0])
    return neuron_ids

def get_cellnum_of_one_celltype(client, celltype='FB4L'):
    q = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId
         """ % (celltype)
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    return len(data_orig)

def get_downstream_of_one_celltype(client, celltype='FB4L'):
    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1:Neuron)
         WHERE n0.type = "%s"
         RETURN DISTINCT n1.type as celltypes_post
         ORDER BY celltypes_post ascending
         """ % celltype
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    return [row[0] for row in data_orig]

def get_total_synnum_btw_two_celltypes(client, celltypes=['FB4L', 'hDeltaB']):
    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1:Neuron)
        WHERE n0.type = '%s' AND n1.type = '%s' 
        RETURN c.weight as weight
         """ % (celltypes[0], celltypes[1])
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    n_tot = 0 if len(data_orig)==0 else np.sum(data_orig, axis=0)[0]
    return n_tot

def wrap(arr, cmin=-180, cmax=180):
    period = cmax - cmin
    arr = arr%period
    arr[arr>=cmax] = arr[arr>=cmax] - period
    arr[arr<cmin] = arr[arr<cmin] + period
    return arr

def connect_to_axons_PFNd2hDeltaB_int(drct, n_g_pfnd, n_g_hdb):
    flag = 0
    if drct == 0:
        index = n_g_pfnd + n_g_hdb
        if ((index >= 5) & (index <= 8)) or ((index >= 15) & (index <= 19)):
            flag = 1
    else:
        if n_g_pfnd == 1:
            n_g_pfnd = 9
        index = n_g_pfnd + n_g_hdb
        if ((index >= 6) & (index <= 10)) or ((index >= 17) & (index <= 21)):
            flag = 1

    return flag

def connect_to_axons_PFNd2hDeltaB(drct, n_g_pfnd, n_g_hdb):
    flag = 0
    if drct == 0:
        index = n_g_pfnd/8. + n_g_hdb/14.
        if ((index >= .4) & (index <= .72)) or ((index >= 1.4) & (index <= 1.75)):
            flag = 1
    else:
        if n_g_pfnd == 1:
            n_g_pfnd = 9
        index = n_g_pfnd/8. + n_g_hdb/14.
        if ((index >= .4) & (index <= .876)) or ((index >= 1.4) & (index <= 1.86)):
            flag = 1

    return flag

def Fglom_to_Fcellid(c, Fglom, celltype):
    Fcellid = []
    if celltype == 'EPG':
        q = """\
                 MATCH (n0:Neuron)
                 WHERE n0.type = '%s'
                 RETURN n0.bodyId, n0.instance
                 ORDER BY n0.bodyId ascending 
                 """ % celltype
        data_orig = np.array(c.fetch_custom(q, format='json')['data'])
        for row, data_ in enumerate(data_orig):
            bodyid, ins = data_[0], data_[1]
            drct, n_g = get_PB_GlomPos_from_instance_1to9(ins)
            i_glom = drct * 9 + n_g - 1
            F_ = Fglom[i_glom]
            Fcellid.append(F_)
    # elif celltype == 'Delta7':

    return np.array(Fcellid)

def Fcellid_to_Fglom(c, Fcellid, celltype, activity_sign=1):
    # Fcellid is a dictionary for Delta7, a list for other cell types ranked as bodyID ascending order
    if celltype == 'Delta7':
        Fs = []
        for i in range(18):
            Fs.append([])
        q = """\
                MATCH (n0:Neuron)
                WHERE n0.type = '%s'
                RETURN n0.bodyId, n0.instance
                ORDER BY n0.bodyId ascending 
                """ % 'Delta7'
        data_orig = np.array(c.fetch_custom(q, format='json')['data'])

        for row, data_ in enumerate(data_orig):
            bodyid, ins = data_[0], data_[1]
            dir_ns = get_PB_GlomPoss_from_Delta7_instance_1to9(ins)
            for dir_n in dir_ns:
                drct, n_g = dir_n
                i_glom = drct * 9 + n_g - 1
                Fs[i_glom].append(Fcellid[int(bodyid)])
        Fs_mean = []
        for i in range(18):
            Fs_mean.append(np.nanmean(Fs[i]))

    elif celltype == 'EPG':
        Fs = []
        for i in range(18):
            Fs.append([])
        q = """\
                MATCH (n0:Neuron)
                WHERE n0.type = '%s'
                RETURN n0.bodyId, n0.instance
                ORDER BY n0.bodyId ascending 
                """ % 'EPG'
        data_orig = np.array(c.fetch_custom(q, format='json')['data'])

        for row, data_ in enumerate(data_orig):
            bodyid, ins = data_[0], data_[1]
            drct, n_g = get_PB_GlomPos_from_instance_1to9(ins)
            i_glom = drct * 9 + n_g - 1
            Fs[i_glom].append(Fcellid[int(bodyid)])
        Fs_mean = []
        for i in range(18):
            Fs_mean.append(np.nanmean(Fs[i]))

    return np.array(Fs_mean) * activity_sign

def Fiddict_to_Fidlist(c, F_ids, celltype):
    F_ids_dict = {}
    q = """\
             MATCH (n0:Neuron)
             WHERE n0.type = '%s'
             RETURN n0.bodyId
             ORDER BY n0.bodyId ascending 
             """ % celltype
    data_orig = np.array(c.fetch_custom(q, format='json')['data'])
    for row, data_ in enumerate(data_orig):
        bodyid = data_[0]
        F_ids_dict[bodyid] = F_ids[row]
    return F_ids_dict

# get Glomeruli Index
def get_PB_GlomPos_from_instance(instance):
    # extract from data
    if instance[-1] == ')':
        GP = instance[-8:-6]
    elif instance[-2] == 'C':
        GP = instance[-5:-3]
    else:
        GP = instance[-2:]
    direction = 0 if GP[0] == 'L' else 1
    num_G = int(GP[1])

    # transform from (987654321, 123456789) to (123456781, 123456781) order
    if direction == 0:  # left: 9-1, 8-2, 7-3, ..., 2-8, 1-1
        num_G_reg = (10 - num_G - 1) % 8 + 1
    else:  # right: 1-2, 2-2, ..., 8-8, 9-1
        num_G_reg = (num_G - 1) % 8 + 1

    return direction, num_G_reg

def get_PB_GlomPos_from_instance_1to9(instance):
    # extract from data
    if instance[-1] == ')':
        GP = instance[-8:-6]
    elif instance[-2] == 'C':
        GP = instance[-5:-3]
    else:
        GP = instance[-2:]
    direction = 0 if GP[0] == 'L' else 1
    num_G = int(GP[1])

    # transform from (987654321, 123456789) to (123456789, 123456789) order
    if direction == 0:  # left: 9-1, 8-2, 7-3, ..., 2-8, 1-9
        num_G_reg = 10 - num_G
    else:  # right: 1-2, 2-2, ..., 8-8, 9-1
        num_G_reg = num_G

    return direction, num_G_reg

def get_PB_GlomPoss_from_Delta7_instance_1to9(instance):
    dir_ns = []
    glomnames = instance.split('_')[1]
    for i in range(int(len(glomnames)/2)):
        glomname = glomnames[i*2:i*2+2]

        direction = 0 if glomname[0] == 'L' else 1
        num_G = int(glomname[1])

        # transform from (987654321, 123456789) to (123456789, 123456789) order
        if direction == 0:  # left: 9-1, 8-2, 7-3, ..., 2-8, 1-9
            num_G_reg = 10 - num_G
        else:  # right: 1-2, 2-2, ..., 8-8, 9-1
            num_G_reg = num_G

        dir_ns.append([direction, num_G_reg])
    return dir_ns

def get_PB_GlomPoss_from_Delta7_instance_1to8(instance):
    dir_ns = []
    glomnames = instance.split('_')[1]
    for i in range(int(len(glomnames)/2)):
        glomname = glomnames[i*2:i*2+2]

        direction = 0 if glomname[0] == 'L' else 1
        num_G = int(glomname[1])

        # transform from (987654321, 123456789) to (123456789, 123456789) order
        if direction == 0:  # left: 9-1, 8-2, 7-3, ..., 2-8, 1-9
            num_G_reg = (10 - num_G - 1) % 8 + 1
        else:  # right: 1-2, 2-2, ..., 8-8, 9-1
            num_G_reg = (num_G - 1) % 8 + 1

        dir_ns.append([direction, num_G_reg])
    return dir_ns

def get_FB_GlomPos_from_instance(instance):
    # extract from data
    num_G = int(instance[-2:])

    # transform from (14-1) to (1-14) order
    num_G_reg = 15 - num_G

    return num_G_reg

def get_hDeltaB_GlomPos_from_instance(instance):
    # extract from data
    num_G = int(instance.split('_')[1])
    return num_G

def get_PFR_FB_GlomPos_from_instance(instance, celltype='PFR'):
    # extract from data, coordination: PB
    if instance[-1] == ')':
        GP = instance[-8:-6]
    elif instance[-1] == 'g':
        GP = instance[-11:-9]
    else:
        GP = instance[-5:-3]
    direction = 0 if GP[0] == 'L' else 1
    num_G = int(GP[1])

    # transform from PB to FB order
    if celltype == 'PFR':
        PB_L_2_FB = [1, 14, 12, 10, 8, 6, 4, 2] # PFR diagram
        PB_R_2_FB = [16, 3, 5, 7, 9, 11, 13, 15] # PFR diagram
    elif celltype == 'PFGs':
        PB_L_2_FB = [1, 15, 13, 11, 9, 7, 5, 3, 2] # PFGs diagram
        PB_R_2_FB = [18, 4, 6, 8, 10, 12, 14, 16, 17] # PFGs diagram
    elif celltype == 'EPG':
        PB_L_2_FB = [1,15,13,11,9,7,5,3]  # EPG diagram
        PB_R_2_FB = [2,4,6,8,10,12,14,16] # EPG diagram
    if direction == 0:  # left
        num_G_reg = PB_L_2_FB[num_G - 1]
    else:  # right
        num_G_reg = PB_R_2_FB[num_G - 1]

    return num_G_reg


# Delta7 project
def comp_F_Delta7_using_F_EPG(c, F_EPGs):
    EPGid2F_dict = {}
    q_EPG = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId, n0.instance
         ORDER BY n0.instance ascending 
         """ % 'EPG'
    data_orig = np.array(c.fetch_custom(q_EPG, format='json')['data'])
    for row, data_ in enumerate(data_orig):  # fill id2index_dict_pre and label0
        bodyid, ins = data_[0], data_[1]
        drct, n_g = get_PB_GlomPos_from_instance_1to9(ins)
        i_EPG = drct * 9 + n_g - 1
        F_EPG = F_EPGs[i_EPG]
        #     print(F_EPG, i_EPG)
        EPGid2F_dict[bodyid] = F_EPG

    Delta7id2F_dict = {}
    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
         WHERE n0.type = '%s' AND n1.type = '%s'
         RETURN n0.bodyId, c.weight AS w, n1.bodyId
         ORDER BY n1.bodyId ascending
         """ % ('EPG', 'Delta7')
    data_EPG2Delta7 = np.array(c.fetch_custom(q, format='json')['data'])

    F_thisDelta7 = 0
    thisDelta7_id = data_EPG2Delta7[0][2]

    for row, data_ in enumerate(data_EPG2Delta7):  # fill id2index_dict_pre and label0
        EPG_id, w, Delta7_id = data_[0], data_[1], data_[2]
        if Delta7_id == thisDelta7_id:
            F_thisDelta7 = F_thisDelta7 + w * EPGid2F_dict[str(EPG_id)]
        else:
            Delta7id2F_dict[thisDelta7_id] = F_thisDelta7
            F_thisDelta7 = w * EPGid2F_dict[str(EPG_id)]
            thisDelta7_id = Delta7_id

        if row == (len(data_EPG2Delta7) - 1):
            Delta7id2F_dict[thisDelta7_id] = F_thisDelta7

    return Delta7id2F_dict

def comp_Delta7_output(c, Delta7id2F_dict):
    Fs = []
    for i in range(18):
        Fs.append([])
    q = """\
             MATCH (n0:Neuron)
             WHERE n0.type = '%s'
             RETURN n0.bodyId, n0.instance
             ORDER BY n0.instance ascending 
             """ % 'Delta7'
    data_orig = np.array(c.fetch_custom(q, format='json')['data'])

    for row, data_ in enumerate(data_orig):  # fill id2index_dict_pre and label0
        bodyid, ins = data_[0], data_[1]
        dir_ns = get_PB_GlomPoss_from_Delta7_instance_1to9(ins)
        for dir_n in dir_ns:
            drct, n_g = dir_n
            i_glom = drct * 9 + n_g - 1
            Fs[i_glom].append(Delta7id2F_dict[int(bodyid)])

    Fs_mean = []
    for i in range(18):
        Fs_mean.append(-np.nanmean(Fs[i]))

    return Fs_mean



# get ANGLE of each Glomerulus
def get_PBAngle_from_PBGlomPos(direction, num_G_reg):
    # num_G_reg ranges from 1 to 8
    if direction == 0:      # 1: 168.75˚, 2: -146.25˚, 3: -123.75˚, ..., 8: 123.7k5
        # angle = -168.75 + (num_G_reg-1) * 45
        angle = 168.75 + (num_G_reg - 1) * 45
    else:                   # 1-22.5˚, 2-67.5˚, ..., 8-337.5˚
        angle = -168.75 + (num_G_reg - 1) * 45

    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle = angle - 360

    return angle

def get_PBAngle_from_PBGlomPos_EPGanatomy(direction, num_G_reg):
    # num_G_reg ranges from 1 to 8
    if direction == 0:      # 1: 168.75˚, 2: -146.25˚, 3: -123.75˚, ..., 8: 123.7k5
        # angle = -168.75 + (num_G_reg-1) * 45
        angle = 0 + (num_G_reg - 1) * 45
    else:                   # 1-22.5˚, 2-67.5˚, ..., 8-337.5˚
        angle = 22.5 + (num_G_reg - 1) * 45

    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle = angle - 360

    return angle

def get_PBAngle_from_PBGlomPos_Delta7anatomy(direction, num_G_reg):
    # num_G_reg ranges from 1 to 8
    if direction == 0:      # 1: 168.75˚, 2: -146.25˚, 3: -123.75˚, ..., 8: 123.7k5
        # angle = -168.75 + (num_G_reg-1) * 45
        angle = 0 + (num_G_reg - 1) * 45 - 11.25
    else:                   # 1-22.5˚, 2-67.5˚, ..., 8-337.5˚
        angle = 22.5 + (num_G_reg - 1) * 45 + 11.25

    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle = angle - 360

    return angle

def get_PBAngle_from_PBGlomPos_PFNvphysiology(direction, num_G_reg):
    # num_G_reg ranges from 1 to 8
    if direction == 0:      # 1: 168.75˚, 2: -146.25˚, 3: -123.75˚, ..., 8: 123.7k5
        # angle = -168.75 + (num_G_reg-1) * 45
        angle = 0 + (num_G_reg - 1) * 45 - 3.275
    else:                   # 1-22.5˚, 2-67.5˚, ..., 8-337.5˚
        angle = 22.5 + (num_G_reg - 1) * 45 + 3.275

    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle = angle - 360

    return angle

def get_PBAngle_from_PBGlomPos_PFNdphysiology(direction, num_G_reg):
    # num_G_reg ranges from 1 to 8
    if direction == 0:      # 1: 168.75˚, 2: -146.25˚, 3: -123.75˚, ..., 8: 123.7k5
        # angle = -168.75 + (num_G_reg-1) * 45
        angle = 0 + (num_G_reg - 1) * 45 - 7.435
    else:                   # 1-22.5˚, 2-67.5˚, ..., 8-337.5˚
        angle = 22.5 + (num_G_reg - 1) * 45 + 7.435

    if angle < -180:
        angle += 360
    elif angle >= 180:
        angle = angle - 360

    return angle

def get_id2angle_dict_PFNv2Delta6B(data_orig, PB_direction, anglefunction=get_PBAngle_from_PBGlomPos):
    id2angle_dict = {}
    id_prev = data_orig[0][-2]
    ws, angles = [], []

    cell_count = 0
    for irow, row in enumerate(data_orig):
        # get position and angle
        PFNv_ins, Delta6B_id, w = row[1], row[-2], int(row[2])
        drct, n_g = get_PB_GlomPos_from_instance(PFNv_ins)
        # print(PFNv_ins, drct, n_g)
        if drct == PB_direction:
            angle = anglefunction(drct, n_g)
        else:
            continue

        if Delta6B_id != id_prev:
            if len(ws):
                # one neuron finished, first load to id2angle_dict, then create new lists and fill
                angle_neuron = ave_weighted_angles(ws, angles)
                id2angle_dict[id_prev] = angle_neuron
                # print(cell_count, id_prev, angle_neuron, ws, angles)
                ws, angles = [], []
                cell_count = cell_count + 1

        id_prev = Delta6B_id
        ws.append(w)
        angles.append(angle)

    # finish the last neuron
    angle_neuron = ave_weighted_angles(ws, angles)
    id2angle_dict[id_prev] = angle_neuron

    return id2angle_dict

def get_id2angle_dict_PFNd2hDeltaB_AxonsOnly(data_orig, PB_direction, anglefunction=get_PBAngle_from_PBGlomPos):
    id2angle_dict = {}
    id_prev = data_orig[0][-2]
    ws, angles = [], []

    cell_count = 0
    for irow, row in enumerate(data_orig):
        # get position and angle
        PFNd_ins, hDeltaB_id, hDeltaB_ins, w = row[1], row[-2], row[-1], int(row[2])
        n_g_hdb = get_hDeltaB_GlomPos_from_instance(hDeltaB_ins)
        drct, n_g = get_PB_GlomPos_from_instance(PFNd_ins)

        if connect_to_axons_PFNd2hDeltaB(drct, n_g, n_g_hdb):
            if drct == PB_direction:
                angle = anglefunction(drct, n_g)
            else:
                continue

            if hDeltaB_id != id_prev:
                if len(ws):
                    # one neuron finished, first load to id2angle_dict, then create new lists and fill
                    angle_neuron = ave_weighted_angles(ws, angles)
                    id2angle_dict[id_prev] = angle_neuron
                    # print(cell_count, id_prev, angle_neuron, ws, angles)
                    ws, angles = [], []
                    cell_count = cell_count + 1

            id_prev = hDeltaB_id
            ws.append(w)
            angles.append(angle)

    # finish the last neuron
    angle_neuron = ave_weighted_angles(ws, angles)
    id2angle_dict[id_prev] = angle_neuron

    return id2angle_dict

def plot_PB_id2angle(client, celltype, ax=False, anglefunction=get_PBAngle_from_PBGlomPos, c=ph.blue, prefix='supp fig-PB_id2angle-', save=True, ms=5, fs=5):
    # extract data
    q = """\
     MATCH (n0:Neuron)
     WHERE n0.type = '%s'
     RETURN n0.bodyId, n0.instance
     ORDER BY n0.instance ascending
     """ % (celltype)
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])

    # organize data
    ids = []
    thetas = []

    # flip_leftpb: flip left pb index, from 1-9 to 9-1
    instances = data_orig[:, 1]
    left_index = np.zeros(len(instances))
    for i_ in range(len(instances)):
        left_index[i_] = ('_L' in instances[i_])
    left_index = left_index.astype(bool)
    data_orig[left_index] = data_orig[left_index][::-1]

    for irow, row in enumerate(data_orig):
        n_id, n_ins = row[0], row[-1]
        drct, n_g = get_PB_GlomPos_from_instance(n_ins)
        theta = anglefunction(drct, n_g)
        ids.append(n_id)
        thetas.append(theta)
    # return ids, thetas

    # plot
    ph.set_fontsize(fs)
    N = len(thetas)
    if not ax:
        _ = plt.figure(1, (.064 * N, .8))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    x = np.arange(N)
    ax.plot(x, thetas, ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.6, xlim=[-.5, N - .5], xticks=np.arange(0, N), ylim=[-185, 185], yticks=np.arange(-180, 181, 90))
    _ = ax.set_xticklabels('')

    if save:
        ph.save(prefix + celltype, exts=['pdf'])

def plot_PB_id2angle_using_dict(client, celltype, id2angle_dict, ax=False, c=ph.blue, prefix='supp fig-PB_id2angle-', save=True, ms=4, fs=5):
    # extract data
    q = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId, n0.instance
         ORDER BY n0.instance descending
         """ % (celltype)
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])

    ids = []
    thetas = []
    for irow, row in enumerate(data_orig):
        n_id = row[0]
        ids.append(n_id)
        thetas.append(id2angle_dict[n_id])

    # plot
    ph.set_fontsize(fs)
    N = len(thetas)
    if not ax:
        _ = plt.figure(1, (.064 * N, .8))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    x = np.arange(N)
    ax.plot(x, thetas, ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.5, xlim=[-.5, N - .5], xticks=np.arange(0, N), ylim=[-185, 185], yticks=np.arange(-180, 181, 90))
    _ = ax.set_xticklabels('')

    if save:
        ph.save(prefix + celltype, exts=['pdf'])

def plot_id2angle(thetas, ids, ax=False, c=ph.blue, prefix='supp fig-PB_id2angle-', celltype='celltype', save=True, ms=5, fs=5):

    ph.set_fontsize(fs)
    N = len(thetas)
    if not ax:
        _ = plt.figure(1, (.064 * N, .8))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    x = np.arange(N)
    ax.plot(x, thetas, ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xlim=[-.5, N - .5], ylim=[-181, 181],
                     yticks=np.arange(-180, 181, 90))

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation='90')

    if save:
        ph.save(prefix + celltype, exts=['pdf'])

def hist_angle_offset(diff_angles, ws, ax=False, fig_size=(2,2), ylim=[0,45000], yticks=[0,20000,40000],
                      lw=2, lw_dash=1, binnum=36, fs=5, prefix='hist', save=False, ):
    # organize data
    y = np.zeros(binnum)
    bins = np.linspace(-180, 180, binnum + 1)
    bin_edges = (bins[1:] + bins[:-1]) / 2
    inds = np.digitize(diff_angles, bins)
    for i, ind in enumerate(inds):
        y[ind - 1] += ws[i]

    # plot
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    ax.plot(bin_edges, y, lw=lw, c='black', zorder=2)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.5, xlim=[-181, 181], xticks=np.arange(-180, 181, 90),
                     ylim=ylim, yticks=yticks)
    for i_ in np.arange(-135, 136, 45):
        ax.axvline(i_, ls='--', color=ph.grey2, lw=lw_dash, zorder=1)

    ax.set_xlabel('offset in PB (deg)')
    ax.set_ylabel('Cell count')
    mean = np.mean(diff_angles)
    std = stats.circstd(diff_angles, 180, -180)
    sem = std / np.sqrt(len(diff_angles))
    # _ = ax.set_title('average center = %.1f deg' % ave_weighted_angles(y, bin_edges))
    _ = ax.set_title('%.1f deg, +- %.1f deg' % (mean, sem))
    print('%.1f deg, +- %.1f deg' % (mean, sem))

    if save:
        ph.save(prefix, exts=['pdf'])


# Connectivity Matrix
def get_cmat(c, n0type='FB6A', n1type='FB5G', labeltype='instance', flip_leftpb=True, projPre_pb2fb_pfr=False,
             projPost_pb2fb_pfr=False, **kwargs):
    q_pre = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId, n0.instance
         ORDER BY n0.instance ascending 
         """ % n0type

    q_post = """\
         MATCH (n1:Neuron)
         WHERE n1.type = '%s'
         RETURN n1.bodyId, n1.instance
         ORDER BY n1.instance ascending 
         """ % n1type

    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
         WHERE n0.type = '%s' AND n1.type = '%s'
         RETURN n0.bodyId, n0.instance, c.weight AS w, n1.bodyId, n1.instance
         ORDER BY w ascending
         """ % (n0type, n1type)

    labeltype_ind = 1 if labeltype == 'instance' else 0
    # Generate dictionary for pre and post bodyid
    data_orig = np.array(c.fetch_custom(q_pre, format='json')['data'])
    id2index_dict0, ncol, label0 = {}, len(data_orig), []
    if flip_leftpb:                                 # flip left pb index, from 1-9 to 9-1
        instances = data_orig[:, 1]
        left_index = np.zeros(len(instances))
        for i_ in range(len(instances)):
            left_index[i_] = ('_L' in instances[i_])
        left_index = left_index.astype(bool)
        data_orig[left_index] = data_orig[left_index][::-1]
    if projPre_pb2fb_pfr:
        instances = data_orig[:, 1]
        pfr_fbGlomN = []
        for i_ in range(len(instances)):
            pfr_fbGlomN.append(get_PFR_FB_GlomPos_from_instance(instances[i_], **kwargs))
        pfr_SortIndex = np.argsort(pfr_fbGlomN)
        data_orig = data_orig[pfr_SortIndex]
    for row, data_ in enumerate(data_orig):         # fill id2index_dict_pre and label0
        bodyid = data_[0]
        id2index_dict0[bodyid] = row
        label0.append(data_[labeltype_ind])

    data_orig = np.array(c.fetch_custom(q_post, format='json')['data'])
    id2index_dict1, nrow, label1 = {}, len(data_orig), []
    if flip_leftpb:                                 # flip left pb index, from 1-9 to 9-1
        instances = data_orig[:, 1]
        left_index = np.zeros(len(instances))
        for i_ in range(len(instances)):
            left_index[i_] = ('_L' in instances[i_])
        left_index = left_index.astype(bool)
        data_orig[left_index] = data_orig[left_index][::-1]
    if projPost_pb2fb_pfr:
        instances = data_orig[:, 1]
        pfr_fbGlomN = []
        for i_ in range(len(instances)):
            pfr_fbGlomN.append(get_PFR_FB_GlomPos_from_instance(instances[i_], **kwargs))
        pfr_SortIndex = np.argsort(pfr_fbGlomN)
        data_orig = data_orig[pfr_SortIndex]

    for row, data_ in enumerate(data_orig):         # fill id2index_dict1 and label1
        bodyid = data_[0]
        id2index_dict1[bodyid] = row
        label1.append(data_[labeltype_ind])

    # Generate Connection Matrix
    cmat = np.zeros((nrow, ncol))
    data_orig = np.array(c.fetch_custom(q, format='json')['data'])
    for data_ in data_orig:
        i0 = id2index_dict0[data_[0]]
        i1 = id2index_dict1[data_[3]]
        w = data_[2]
        cmat[i1, i0] = w

    return label0, label1, cmat

def get_cmat_cellbodyid_rank(c, n0type='FB6A', n1type='FB5G', **kwargs):
    q_pre = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId
         ORDER BY n0.bodyId ascending 
         """ % n0type

    q_post = """\
         MATCH (n1:Neuron)
         WHERE n1.type = '%s'
         RETURN n1.bodyId
         ORDER BY n1.bodyId ascending 
         """ % n1type

    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
         WHERE n0.type = '%s' AND n1.type = '%s'
         RETURN n0.bodyId, c.weight AS w, n1.bodyId
         ORDER BY w ascending
         """ % (n0type, n1type)

    # Generate dictionary for pre and post bodyid
    data_orig = np.array(c.fetch_custom(q_pre, format='json')['data'])
    id2index_dict0, ncol, label0 = {}, len(data_orig), []
    for row, data_ in enumerate(data_orig):         # fill id2index_dict_pre and label0
        bodyid = data_[0]
        id2index_dict0[bodyid] = row
        label0.append(bodyid)

    data_orig = np.array(c.fetch_custom(q_post, format='json')['data'])
    id2index_dict1, nrow, label1 = {}, len(data_orig), []
    for row, data_ in enumerate(data_orig):         # fill id2index_dict1 and label1
        bodyid = data_[0]
        id2index_dict1[bodyid] = row
        label1.append(bodyid)

    # Generate Connection Matrix
    cmat = np.zeros((nrow, ncol))
    data_orig = np.array(c.fetch_custom(q, format='json')['data'])
    for data_ in data_orig:
        i0 = id2index_dict0[data_[0]]
        i1 = id2index_dict1[data_[2]]
        w = data_[1]
        cmat[i1, i0] = w

    return label0, label1, cmat

def get_cmat_test2(c, n0type='PFNd', n1type='hDeltaB', labeltype='instance', flip_leftpb=True):
    q_pre = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId, n0.instance
         ORDER BY n0.instance ascending 
         """ % n0type

    q_post = """\
         MATCH (n1:Neuron)
         WHERE n1.type = '%s'
         RETURN n1.bodyId, n1.instance
         ORDER BY n1.instance ascending 
         """ % n1type

    q = """\
         MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
         WHERE n0.type = '%s' AND n1.type = '%s'
         RETURN n0.bodyId, n0.instance, c.weight AS w, n1.bodyId, n1.instance
         ORDER BY w ascending
         """ % (n0type, n1type)

    labeltype_ind = 1 if labeltype == 'instance' else 0
    # Generate dictionary for pre and post bodyid
    data_orig = np.array(c.fetch_custom(q_pre, format='json')['data'])
    id2index_dict0, ncol, label0 = {}, len(data_orig), []
    if flip_leftpb:                                 # flip left pb index, from 1-9 to 9-1
        instances = data_orig[:, 1]
        left_index = np.zeros(len(instances))
        for i_ in range(len(instances)):
            left_index[i_] = ('_L' in instances[i_])
        left_index = left_index.astype(bool)
        data_orig[left_index] = data_orig[left_index][::-1]

    for row, data_ in enumerate(data_orig):         # fill id2index_dict_pre and label0
        bodyid = data_[0]
        id2index_dict0[bodyid] = row
        label0.append(data_[labeltype_ind])

    data_orig = np.array(c.fetch_custom(q_post, format='json')['data'])
    id2index_dict1, nrow, label1 = {}, len(data_orig), []
    if flip_leftpb:                                 # flip left pb index, from 1-9 to 9-1
        instances = data_orig[:, 1]
        left_index = np.zeros(len(instances))
        for i_ in range(len(instances)):
            left_index[i_] = ('_L' in instances[i_])
        left_index = left_index.astype(bool)
        data_orig[left_index] = data_orig[left_index][::-1]

    for row, data_ in enumerate(data_orig):         # fill id2index_dict1 and label1
        bodyid = data_[0]
        id2index_dict1[bodyid] = row
        label1.append(data_[labeltype_ind])

    # Generate Connection Matrix
    cmat = np.zeros((nrow, ncol))
    data_orig = np.array(c.fetch_custom(q, format='json')['data'])
    for data_ in data_orig:
        PFNd_ins, hDeltaB_ins = data_[1], data_[-1]
        n_g_hdb = get_hDeltaB_GlomPos_from_instance(hDeltaB_ins)
        drct, n_g = get_PB_GlomPos_from_instance(PFNd_ins)
        if not connect_to_axons_PFNd2hDeltaB(drct, n_g, n_g_hdb):
            i0 = id2index_dict0[data_[0]]
            i1 = id2index_dict1[data_[3]]
            w = data_[2]
            cmat[i1, i0] = w

    return label0, label1, cmat

def get_cmat_cell2celltype(c, n0type='PFNd', n1types=['hDeltaB']):
    q_pre = """\
         MATCH (n0:Neuron)
         WHERE n0.type = '%s'
         RETURN n0.bodyId, n0.instance
         ORDER BY n0.instance ascending 
         """ % n0type

    # Generate dictionary for pre and post bodyid
    nrow = len(n1types)
    data_orig = np.array(c.fetch_custom(q_pre, format='json')['data'])
    id2index_dict0, ncol = {}, len(data_orig)
    for row, data_ in enumerate(data_orig):         # fill id2index_dict_pre and label0
        bodyid = int(data_[0])
        id2index_dict0[bodyid] = row

    # Generate Connection Matrix
    cmat = np.zeros((nrow, ncol))
    for row, n1type in enumerate(n1types):
        q = """\
             MATCH (n0:Neuron) -[c:ConnectsTo]-> (n1 :Neuron)
             WHERE n0.type = '%s' AND n1.type = '%s'
             RETURN n0.bodyId, c.weight AS w, n1.bodyId
             ORDER BY w ascending
             """ % (n0type, n1type)
        data_orig = np.array(c.fetch_custom(q, format='json')['data'])
        ws_thisrow = np.zeros(ncol)
        for data_ in data_orig:
            i0 = id2index_dict0[data_[0]]
            w = data_[1]
            ws_thisrow[i0] = ws_thisrow[i0] + w
        cmat[row, :] = ws_thisrow

    return cmat

def plot_cmat(cmat, label0, label1, fig_size=(4.1, 4), cm=plt.cm.viridis, flip_postsyn=False, flip_presyn=False,
              show_labels=True, rotation=90, fs=5, vmax=0):
    nrow, ncol = np.shape(cmat)
    if flip_presyn:
        # cmat = cmat[::-1,:]
        cmat = cmat[:, ::-1]
        label0 = label0[::-1]
    if flip_postsyn:
        # cmat = cmat[:,::-1]
        cmat = cmat[::-1, :]
        label1 = label1[::-1]

    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[40,1])
    ax = plt.subplot(gs[0, 0])
    img = ax.pcolormesh(np.arange(ncol + 1), np.arange(nrow + 1), cmat, cmap=cm)
    ax.set_xticks(np.arange(ncol) + .5)
    ax.set_yticks(np.arange(nrow) + .5)
    ax.tick_params('both', length=1, width=.5, which='major')
    if show_labels:
        _ = ax.set_xticklabels(label0, rotation=rotation, ha='center', fontsize=fs)
        _ = ax.set_yticklabels(label1, fontsize=fs)
    else:
        _ = ax.set_xticklabels('')
        _ = ax.set_yticklabels('')
    ax.xaxis.tick_top()

    if not vmax:
        vmax = np.max(cmat)
    img.set_clim(vmax=vmax)
    img.set_clim(vmin=0)
    ph.plot_colormap(plt.subplot(gs[0, 1]), colormap=cm, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                     yticklabels=['0', '%.2f' % vmax],
                     ylabel='snp#', label_rotation=0, label_pos="right")




# 3Rotation
def create_3d_rotation_axis(theta=pi * 0.25, phi=pi * 0.25, varphi=pi * 0):
    nn = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    xn_ = [np.cos(varphi), np.sin(varphi), 0]
    xn = xn_ - np.dot(xn_, nn) * nn
    xn /= np.sqrt((xn ** 2).sum())
    yn = np.cross(nn, xn)
    return nn, xn, yn

def plot_neurons_and_rois(client, neuron_ids=[634608015], roi_names=['FB', 'PB'], ax=False, neuron_colors=[ph.red],
                          fig_size=(3,3), rotation_3d=[1.6, 1.6, 0], color_roi=ph.grey4, alpha_roi=.1, alpha_neuron=0.6,
                          xlim=[9000,-9000], ylim=[-9000,9000], offsets=[0, 0], ):
    # The first roi will be used to center
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    # define three rotation angles, rad
    theta, phi, varphi = rotation_3d

    # calculate the unit vector: nn-projection dimention, xn & yn: diagonal of projected plane
    nn = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    xn_ = [np.cos(varphi), np.sin(varphi), 0]
    xn = xn_ - np.dot(xn_, nn) * nn
    xn /= np.sqrt((xn ** 2).sum())
    yn = np.cross(nn, xn)

    for i_roi, roi_name in enumerate(roi_names):
        roi = client.fetch_roi_mesh(roi_name)                    # extract the 3d coordinates of the neuron and ROIs
        roi_3d = np.array([np.array(row.split(' ')[1:]).astype(float) for row in str(roi).split('\\n')
                          if row.startswith('v')])
        roi_2d = np.array([np.dot(roi_3d[:, :3], xn), np.dot(roi_3d[:, :3], yn)]).T

        if i_roi == 0:                                      # centroid of the plot
            xc = (np.min(roi_2d[:, 0]) + np.max(roi_2d[:, 0])) / 2.
            yc = (np.min(roi_2d[:, 1]) + np.max(roi_2d[:, 1])) / 2.
            xc = xc + offsets[0]
            yc = yc + offsets[1]

        ax.scatter(roi_2d[:, 0] - xc, roi_2d[:, 1] - yc, s=1, marker='o', edgecolors='none', facecolors=color_roi,
                   alpha=alpha_roi, zorder=1)
        ax.scatter(roi_2d[:, 0] - xc, roi_2d[:, 1] - yc, s=1, marker='o', edgecolors='none', facecolors=color_roi, alpha=alpha_roi, zorder=1)

    for i_neuron, neuron_id in enumerate(neuron_ids):
        sk = client.fetch_skeleton(neuron_id, format='swc')      # extract the 3d coordinates of the neuron and ROIs
        neuron_3d = np.array([np.array(row.split(' ')[2:6]).astype(float) for row in sk.split('\n')
                          if len(row) and row[0].isdigit()])
        neuron_2d = np.array([np.dot(neuron_3d[:, :3], xn), np.dot(neuron_3d[:, :3], yn)]).T
        color_ = neuron_colors[i_neuron%len(neuron_colors)]
        ax.scatter(neuron_2d[:, 0] - xc, neuron_2d[:, 1] - yc, s=neuron_3d[:, 3] / 40., marker='o', edgecolors='none',
                   facecolors=color_, alpha=alpha_neuron, zorder=2)

    ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, [], lw=1, xlim=xlim, ylim=ylim, xticks=[], yticks=[])

def plot_synapses_btw_2neurons(client, neuron_ids=[1225640135, 818796911], roi_names=['FB'], ax=False, fig_size=(3,3),
                               neuron_colors=[ph.green, ph.purple], rotation_3d=[1.6, 1.6, 0],
                               color_synapse=ph.brown, s_synapse=2, alpha_synapse=1, color_roi=ph.grey4, alpha_roi=.1,
                               s_factor_neuron=40., alpha_neuron=0.6, xlim=[9000,-9000], ylim=[-9000,9000],
                               offsets=[0, 0], fs=8):
    # The first roi will be used to center
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    # define three rotation angles, rad
    theta, phi, varphi = rotation_3d

    # calculate the unit vector: nn-projection dimention, xn & yn: diagonal of projected plane
    nn = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    xn_ = [np.cos(varphi), np.sin(varphi), 0]
    xn = xn_ - np.dot(xn_, nn) * nn
    xn /= np.sqrt((xn ** 2).sum())
    yn = np.cross(nn, xn)

    for i_roi, roi_name in enumerate(roi_names):
        roi = client.fetch_roi_mesh(roi_name)                    # extract the 3d coordinates of the neuron and ROIs
        roi_3d = np.array([np.array(row.split(' ')[1:]).astype(float) for row in str(roi).split('\\n')
                          if row.startswith('v')])
        roi_2d = np.array([np.dot(roi_3d[:, :3], xn), np.dot(roi_3d[:, :3], yn)]).T

        if i_roi == 0:                                      # centroid of the plot
            xc = (np.min(roi_2d[:, 0]) + np.max(roi_2d[:, 0])) / 2.
            yc = (np.min(roi_2d[:, 1]) + np.max(roi_2d[:, 1])) / 2.
            xc = xc + offsets[0]
            yc = yc + offsets[1]

        ax.scatter(roi_2d[:, 0] - xc, roi_2d[:, 1] - yc, s=1, marker='o', edgecolors='none', facecolors=color_roi,
                   alpha=alpha_roi, zorder=1)
        ax.scatter(roi_2d[:, 0] - xc, roi_2d[:, 1] - yc, s=1, marker='o', edgecolors='none', facecolors=color_roi, alpha=alpha_roi, zorder=1)

    for i_neuron, neuron_id in enumerate(neuron_ids):
        sk = client.fetch_skeleton(neuron_id, format='swc')      # extract the 3d coordinates of the neuron and ROIs
        neuron_3d = np.array([np.array(row.split(' ')[2:6]).astype(float) for row in sk.split('\n')
                          if len(row) and row[0].isdigit()])
        neuron_2d = np.array([np.dot(neuron_3d[:, :3], xn), np.dot(neuron_3d[:, :3], yn)]).T
        color_ = neuron_colors[i_neuron%len(neuron_colors)]
        ax.scatter(neuron_2d[:, 0] - xc, neuron_2d[:, 1] - yc, s=neuron_3d[:, 3] / s_factor_neuron, marker='o', edgecolors='none',
                   facecolors=color_, alpha=alpha_neuron, zorder=2)

    # plot synapses
    q = """
        MATCH (a:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s_pre:Synapse)-[:SynapsesTo]->(s_post:Synapse)<-[:Contains]-(:SynapseSet)<-[:Contains]-(b:Neuron)
        WHERE a.bodyId = %i AND b.bodyId = %i
        RETURN DISTINCT s_pre.location
        """ % (neuron_ids[0], neuron_ids[1])
    data_orig = np.array(client.fetch_custom(q, format='json')['data'])
    synapses_x = []
    synapses_y = []
    for row in data_orig:
        synapse_3d = row[0]['coordinates']
        synapse_2d = np.array([np.dot(synapse_3d, xn), np.dot(synapse_3d, yn)]).T
        synapses_x.append(synapse_2d[0])
        synapses_y.append(synapse_2d[1])
    ax.scatter(synapses_x - xc, synapses_y - yc, s=s_synapse, marker='o', edgecolors='none',
               facecolors=color_synapse, alpha=alpha_synapse, zorder=3)

    ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, [], lw=1, xlim=xlim, ylim=ylim, xticks=[], yticks=[])
    x_text = xlim[0] + (xlim[-1] - xlim[0]) / 2.
    y_text = ylim[0] - (ylim[-1] - ylim[0]) / 50.
    ax.text(x_text, y_text, 'synapse # = %i' % len(data_orig), ha='center', va='top', fontsize=fs)



# synapses location analysis


































