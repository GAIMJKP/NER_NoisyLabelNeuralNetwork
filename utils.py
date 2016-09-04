__author__ = 'esthervandenberg'

import numpy as np

def random_select(P):
    # takes a sequence P of real positive numbers,
    # randomly selects an element p and return its index i

    # construct scale of classes
    accum_P = [P[0]]
    for i in range(1, len(P)):
        accum_P.append(accum_P[i-1]+P[i])

    # guess a number and find associated class
    i = random.uniform(0.01, .99)

    find_interval = list(accum_P)
    find_interval.append(i)
    find_interval.sort()

    i_idx = find_interval.index(i)
    right_bound = find_interval[i_idx+1]

    label = accum_P.index(right_bound)

    """
    print(accum_P)
    print(i)
    print(find_interval)
    print(i_idx)
    print(right_bound) ###
    print(label)
    """

    return label

###

def get_zt(labels):
    nr_instances = labels.shape[0]
    return [list(labels[t]).index(1) for t in range(nr_instances)]

def get_reverse_zt(c, start_from_0=True):
    nr_instances = len(c)
    nr_classes = max(c)+1
    reverse_zt = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        if start_from_0:
            labelind = c[t]
        else:
            labelind = c[t]-1
        reverse_zt[t,labelind] = 1
    return reverse_zt

###
###

def dist(A, B):
    return abs(get_frob_norm(A) - get_frob_norm(B))

def get_frob_norm(A):
    return np.sqrt(np.trace(np.dot(np.transpose(A), A)))

###

def make_uni_noisy(labels, p=0.1):
    nr_instances = labels.shape[0]
    nr_classes = labels.shape[1]

    # construct theta based on p
    zt = get_zt(labels)
    th = np.zeros([nr_classes]*2)
    for i in range(nr_classes):
        th[i] = [p/(nr_classes-1)]*10
        th[i,i] = 1-p/(nr_classes-1)*9

    # generate noisy labels using theta
    noisy_labels = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        P = th[zt[t]]
        label = random_select(P)
        noisy_labels[t, label] = 1

    return noisy_labels

###

def frac_from_lev(noise_level):
    frac = float(noise_level[0] + '.' + noise_level[1])
    return frac

def lev_from_frac(frac):
    # 0.00 to 000
    str_from_flt = str(frac)
    str_wo_dot = str_from_flt.replace('.', '')
    return str_wo_dot


###