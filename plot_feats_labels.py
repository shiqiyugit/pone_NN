import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", type=str, default=None,
                    dest="input_files", help="name for input files")
parser.add_argument("-o", "--output_dir", type=str, default=None,
                    dest="output", help="name for output directory")
parser.add_argument("-n", "--file_number", type=int, default=None,
                    required=False, dest="n",
                    help="includes first n sorted files from input")

args = parser.parse_args()
files = args.input_files
output = args.output
if args.n:
    n = args.n

if len(files) > 1:
    files = sorted(glob.glob(files))
    
    if n:
        files = files[:n]
    
    energies_all = []
    zenith_all = []
    azimuth_all = []
    pulse_charge_all = []
    pulse_time_all = []
    dom_index_all = []
    dir_x = []
    dir_y = []
    dir_z = []
    vtx_x = []
    vtx_y = []
    vtx_z = []

    for file in files:
        file_in = h5py.File(file, "r")

        features = file_in['features']
        labels = file_in['labels']

        dom_index = np.asarray(features['dom_index'])
        pulse_time = np.asarray(features['pulse_time'])
        pulse_charge = np.asarray(features['pulse_charge'])

        energies = np.asarray(labels['energy'])
        zenith = np.asarray(labels['zenith'])
        azimuth = np.asarray(labels['azimuth'])
        dx = np.asarray(labels['dir_x'])
        dy = np.asarray(labels['dir_y'])
        dz = np.asarray(labels['dir_z'])
        vx = np.asarray(labels['vtx_x'])
        vy = np.asarray(labels['vtx_y'])
        vz = np.asarray(labels['vtx_z'])

        for e in energies:
            energies_all.append(e)
        for z in zenith:
            zenith_all.append(z)
        for a in azimuth:
            azimuth_all.append(a)
        for p in pulse_charge:
            pulse_charge_all.append(p)
        for t in pulse_time:
            pulse_time_all.append(t)
        for x in dx:
            dir_x.append(x)
        for y in dy:
            dir_y.append(y)
        for z in dz:
            dir_z.append(z)
        for x in vx:
            vtx_x.append(x)
        for y in vy:
            vtx_y.append(y)
        for z in vz:
            vtx_z.append(z)        
        for d in dom_index:
            dom_index_all.append(d)
        file_in.close()
    
    energies = energies_all
    zenith = zenith_all
    azimuth = azimuth_all 
    pulse_charge = pulse_charge_all 
    pulse_time = pulse_time_all 
    dom_index = dom_index_all

elif len(files) == 1:

    energies = []
    zenith = []
    azimuth = []
    pulse_charge = []
    pulse_time = []
    dom_index = []
    dir_x = []
    dir_y = []
    dir_z = []
    vtx_x = []
    vtx_y = []
    vtx_z = []

    file_in = h5py.File(files[0], "r")

    features = file_in['features']
    labels = file_in['labels']

    # if "reco" in file_in.keys():
    #     print("reco: True")
    # else:
    #     print("reco: False")

    # for i in file_in.keys():
    #     print(i)
    
    # summary = file_in['summary']

    dom_index = np.asarray(features['dom_index'])
    pulse_time = np.asarray(features['pulse_time'])
    pulse_charge = np.asarray(features['pulse_charge'])
    pmt_index = np.asarray(features['pmt_index'])

    # no_hits_dom = np.asarray(summary['no_hits_dom'])
    # no_hits_pmt = np.asarray(summary['no_hits_pmt'])
    # energy_dom = np.asarray(summary['energy_dom'])
    # energy_pmt = np.asarray(summary['energy_pmt'])


    energies = np.asarray(labels['energy'])
    zenith = np.asarray(labels['zenith'])
    azimuth = np.asarray(labels['azimuth'])
    dir_x = np.asarray(labels['dir_x'])
    dir_y = np.asarray(labels['dir_y'])
    dir_z = np.asarray(labels['dir_z'])
    vtx_x = np.asarray(labels['vtx_x'])
    vtx_y = np.asarray(labels['vtx_y'])
    vtx_z = np.asarray(labels['vtx_z'])
    
    file_in.close()

else:
    raise RuntimeError("No files specified")

bins_z = np.arange(-1.0,1.2,.2)
bins_a = np.arange(0,2*np.pi+.3,.75)

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(np.cos(zenith), bins=bins_z)
ax.set_xlabel("cos(zenith)")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/zenith_dist.png")
del fig, ax

print("zenith done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(azimuth, bins=bins_a)
ax.set_xlabel("azimuth")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/azimuth_dist.png")

print("azimuth done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(energies,)
ax.set_xlabel("energy")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/energy_dist.png")

print("energy done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(dir_x,)
ax.set_xlabel("dir_x")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/dir_x_dist.png")
del fig, ax

print("dir x done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(dir_y,)
ax.set_xlabel("dir_y")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/dir_y_dist.png")
del fig, ax

print("dir y done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(dir_z,)
ax.set_xlabel("dir_z")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/dir_z_dist.png")
del fig, ax

print("dir z done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(vtx_x,)
ax.set_xlabel("vtx_x")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/vtx_x_dist.png")
del fig, ax

print("vtx x done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(vtx_y,)
ax.set_xlabel("vtx_y")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/vtx_y_dist.png")
del fig, ax

print("vtx y done")

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(vtx_z,)
ax.set_xlabel("vtx_z")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/vtx_z_dist.png")
del fig, ax

print("vtx z done")

doms_all = []
for i in range(len(dom_index)):
    for j in range(len(dom_index[i])):
        doms_all.append(dom_index[i][j])

fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)
ax.hist(doms_all,)
ax.set_xlabel("doms")
ax.set_ylabel("Unweighted Counts")
fig.savefig(output + "/dom_ind_dist.png")
del fig, ax

print("DOMs done")
