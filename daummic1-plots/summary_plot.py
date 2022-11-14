import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str, nargs='+',
    dest="input_files",
    help="paths to input files (absolute or relative)")

args = parser.parse_args()
files = args.input_files

energies_all = []
zenith_all = []
azimuth_all = []
pulse_charge_all = []
hits_dom = []
#frag_dom = []
hits_pmt = []
#frag_pmt = []
e_dom = []
#frag_e_dom = []
e_pmt = []
#frag_e_pmt = []

for file in files:
    file_in = h5py.File(file, "r")

    features = file_in['features']
    labels = file_in['labels']
    summary = file_in['summary']

    dom_index = np.asarray(features['dom_index'])
    pulse_time = np.asarray(features['pulse_time'])
    pulse_charge = np.asarray(features['pulse_charge'])
    pmt_index = np.asarray(features['pmt_index'])

    no_hits_dom = np.asarray(summary['no_hits_dom'])
    no_hits_pmt = np.asarray(summary['no_hits_pmt'])
    energy_dom = np.asarray(summary['energy_dom'])
    energy_pmt = np.asarray(summary['energy_pmt'])


    energies = np.asarray(labels['energy'])
    zenith = np.asarray(labels['zenith'])
    azimuth = np.asarray(labels['azimuth'])

    # this is NOT the most efficient way to do this, but anyway...
    for e in energies:
        energies_all.append(e)
    for z in zenith:
        zenith_all.append(z)
    for a in azimuth:
        azimuth_all.append(a)
    for p in pulse_charge:
        pulse_charge_all.append(p)
    for d in no_hits_dom:
        #frag_dom.append(d)
        for n in d:
            hits_dom.append(n)
    for p in no_hits_pmt:
        #frag_pmt.append(p)
        for n in p:    
            hits_pmt.append(n)
    for e in energy_dom:
        #frag_e_dom.append(e)
        for d in e:
            e_dom.append(d)
    for e in energy_pmt:
        #frag_e_pmt.append(e)
        for d in e:
            e_pmt.append(d)



    file_in.close()



bins_e = np.logspace(3,8,10)
bins_z = np.arange(-1.0,1.2,.2)
bins_a = np.arange(0,2*np.pi+.3,.75)
bins_q = np.arange(0,500,100)



fig, ax = plt.subplots()
fig.set_figheight(4.5)
fig.set_figwidth(7)

bins_hits_dom = np.logspace(.01,5,15)
bins_hits_pmt = np.logspace(.01,5,15)
bins_energy = np.logspace(2,8,20)


a, b, c, im = ax.hist2d(e_dom,hits_dom, bins = [bins_energy,bins_hits_dom],
    cmap=plt.cm.plasma, norm=colors.LogNorm())
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Number of Hits (DOM)")
cbar = plt.colorbar(im, ax=ax, label='Events')
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('plots/hits_per_dom.png')

# ax.hist(no_hits_dom, bins=bins_hits_dom)
