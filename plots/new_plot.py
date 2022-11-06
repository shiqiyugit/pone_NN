import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors
#import Plots

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

for file in files:
    file_in = h5py.File(file, "r")

    features = file_in['features']
    labels = file_in['labels']

    dom_index = np.asarray(features['dom_index'])
    pulse_time = np.asarray(features['pulse_time'])
    pulse_charge = np.asarray(features['pulse_charge'])

    try:
        reco = file_in['reco']
    except:
        pass
    
    try:
        print("reco", reco)
    except:
        pass

    energies = np.asarray(labels['energy'])
    zenith = np.asarray(labels['zenith'])
    azimuth = np.asarray(labels['azimuth'])

    #data = file_in['data']
    #energies = np.asarray(data['energies'])
    #zenith = np.asarray(data['zenith'])
    #azimuth = np.asarray(data['azimuth'])
    #print(energies)
    #print("zenith angles", zenith)
    #print("azmimuth angles", azimuth)

    # this is NOT the most efficient way to do this, but anyway...
    for e in energies:
        energies_all.append(e)
    for z in zenith:
        zenith_all.append(z)
    for a in azimuth:
        azimuth_all.append(a)
    for p in pulse_charge:
        pulse_charge_all.append(p)

    file_in.close()


#print(dom_index)
#print(pulse_time)


bins_e = np.logspace(3,8,10)
bins_z = np.arange(-1.0,1.2,.2)
bins_a = np.arange(0,2*np.pi+.3,.75)
bins_q = np.arange(0,500,100)


no_DOMs = np.zeros(len(dom_index))
for i in range(len(dom_index)):
        no_DOMs[i] = len(np.unique(dom_index[i]))

print(dom_index)

no_hits_DOM = np.zeros(len(pulse_charge))
if len(no_hits_DOM) != len(energies):
    raise RunTimeError("Length of energies (%i) and hits (%i) do not match"%(len(energies_all),len(no_hits_DOM)))





for i in range(len(pulse_charge_all)):
    for j in range(len(pulse_charge_all[i])):
        if len(pulse_charge_all[i]) != 0:
            no_hits_DOM[i][j] = len(pulse_charge_all[i][j])
        else:
            no_hits_DOM[i][j] = 0



# no_hits_all = np.zeros(len(pulse_charge_all))

# if len(no_hits_all) != len(energies_all):
#     raise RuntimeError("Length of energies (%i) and hits per DOM (%i) do not match"%(len(energies), len(no_hits_all)))

# for i in range(len(no_hits_all)):
#     no_hits_all[i] = len(pulse_charge_all[i])

#hit_dist = np.array([])
#energy_dist = np.array([])
#for i in range(0,int(max(energies))):
#    energy_dist = np.append(energy_dist,i)
#    relevant_energies = np.logical_and(energies >= i,energies < i+1)
#    relevant_hits = no_hits[relevant_energies]
#    if len(relevant_hits) == 0:
#        hit_dist = np.append(hit_dist,0)
#    else:
#        hit_dist = np.append(hit_dist,sum(relevant_hits)/len(relevant_hits))
# max_hits = math.ceil(max(no_hits)/100.0)*100
# fig, ax = plt.subplots()
# fig.set_figheight(4.5)
# fig.set_figwidth(7)
# #ax.set_xlabel("Number of Hits per Event")
# #ax.set_ylabel("Unweighted Counts")



# bins_hits = np.logspace(.01,6,7)

# # max_E = math.ceil(max(energies_all)/100.0)*100
# #ax.hist(no_hits, bins=bins_hits)
# #fig.savefig('test_hits')
# # del fig, ax

# fig, ax = plt.subplots()
# fig.set_figheight(4.5)
# fig.set_figwidth(7)



#create bin array
#bins = [[],[]]

# bins_energy = np.logspace(2,8,20)
# a, b, c, im = ax.hist2d(energies_all,no_hits, bins = [bins_energy,bins_hits],
#     cmap=plt.cm.plasma, norm=colors.LogNorm())
# #ax.hist(energies_all, bins=bins_energy)
# ax.set_ylabel("Number of Hits")
# ax.set_xlabel("Energy [GeV]")
# cbar = plt.colorbar(im, ax=ax, label='Events')
# ax.set_xscale('log')
# ax.set_yscale('log')




# ax.clear()
# ax.hist(energies_all,bins=bins_energy)
# ax.set_ylabel("Unweighted Counts")
# ax.set_xlabel("Energy [GeV]")
# ax.set_xscale('log')

    
    
    # #vals, x_edges, y_edges, im = ax2.hist2d(
	# #np.log10(energies), np.log10(charge), #weights=weights_from_calc,                                                                                         
    #     bins=bins,
    # )
    # ax2.plot(e_plot_bins, np.log10(charge_plot_bins),
    #          '--', color='red', label='0.075 PE / GeV')
    # ax2.plot(e_plot_bins, np.log10(charge_cut_bins),
    #          '--', color='blue', label='0.025 PE / GeV')

    # # ax2.legend()
    # sim_cbar = plt.colorbar(im, ax=ax2, label='unweighted Events')
    # # lims = 1E-3, 1E-1                                                                                                                                           
    # # im.set_clim(lims)                                                                                                                                           
    # ax2.set_xlabel('Neutrino Energy log10(GeV)')
    # ax2.set_ylabel('log10(HQTot)')
    # fig2.tight_layout()
    # fig2.savefig('charge_vs_energy.png')







#fig.show()
fig.tight_layout()
fig.savefig('test_2d.png')

#uncomment these when pulse series fixed

#for i in range(4):
#    fig, ax = plt.subplots()
#    fig.set_figheight(4.5)
#    fig.set_figwidth(7)
#    if i == 0:
#        ax.hist(energies_all, bins=bins_e)
#        ax.set_xlabel("Energy [GeV]")
#        ax.set_ylabel("Unweighted Counts")
#        ax.set_yscale('log')
#        ax.set_xscale('log')
#        fig.savefig("plot_test/test_E.png")

#    elif i == 1:
#        ax.hist(np.cos(zenith_all), bins=bins_z)
#        ax.set_xlabel("cos(zenith)")
#        ax.set_ylabel("Unweighted Counts")
#        fig.savefig("plot_test/test_theta.png")

    #elif i == 2:
        #ax.hist(azimuth_all, bins=bins_a)
        #ax.set_xlabel("Azimuth Angle [rad]")
        #ax.set_ylabel("Unweighted Counts")
        #fig.savefig("plot_test/test_phi.png")





#ax.hist(energies_all, bins=bins_e)
#ax.set_xlabel("Energy [GeV]")
#ax.set_ylabel("Unweighted Counts")
#ax.set_yscale('log')
#ax.set_xscale('log')
#fig.savefig("test.png")