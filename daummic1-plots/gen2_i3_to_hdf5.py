#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
#METAPROJECT: simulation/V06-01-00-RC4

# we need icesim because of MuonGun
# icerec could be (for example): #METAPROJECT: icerec/V05-02-02-RC2

import os, sys
import glob
import numpy as np
import h5py
import argparse
from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units
#from icecube import MuonGun, simclasses

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default=None,
                    dest="input_files", help="name for input file",
                    required=True, nargs='+')
parser.add_argument("-o", "--overwrite", type=bool, default=False,
                    dest="overwrite", help="whether or not to overwrite previous files")
parser.add_argument("-p", "--pulse_type", type=str, default='cleaned',
                    dest="pulse_type", help="type of pulseseries to use")
args = parser.parse_args()

if len(args.input_files) > 1:
    input_files = sorted(glob.glob(args.input_files))
else:
    input_files = args.input_files
#input_files = sorted(glob.glob("/mnt/home/daummic1/IceCube/gen2_MC/training/0000000-0000999/*.i3.bz2"))
#GenerateSingleMuons_*_photonprop_daqSim_noise_ON.i3.gz"))
#/mnt/research/IceCube/willey/Upgrade_RNN/pre_processed_data/140022/upgrade_genie_step4_140022_000???.i3.zst"))
overwrite = args.overwrite
pulse_type = str.lower(args.pulse_type)

def load_geometry(filename): # Gets geometry from specific geometry file
    
    geo_file = dataio.I3File(filename)
    while geo_file.more():
        frame = geo_file.pop_frame()
        if not (frame.Stop==icetray.I3Frame.Geometry):
            continue
        geometry = frame["I3Geometry"]
        geo_file.close()
        del geo_file
        return geometry
    
    geo_file.close()
    del geo_file
    return None

def read_files(filename_list):
    def track_get_pos(p, length):
        if (not np.isfinite(length)) or (length < 0.) or (length >= p.length):
            return dataclasses.I3Position(np.nan, np.nan, np.nan)
        return dataclasses.I3Position( p.pos.x + length*p.dir.x, p.pos.y + length*p.dir.y, p.pos.z + length*p.dir.z )

    def track_get_time(p, length):
        if (not np.isfinite(length)) or (length < 0.) or (length >= p.length):
            return np.nan
        return p.time + length/p.speed

    weights = []

    features = dict()
    features["dom_index"] = [] # Use DOM indexing for regular simulation
    features["pulse_time"] = []
    features["pulse_charge"] = []
    features["pmt_index"] = []

    labels = dict()
    labels["energy"] = []
    labels["azimuth"] = []
    labels["zenith"] = []
    labels["dir_x"] = []
    labels["dir_y"] = []
    labels["dir_z"] = []
    labels["vtx_x"] = []
    labels["vtx_y"] = []
    labels["vtx_z"] = []

    summary = dict()
    summary["no_hits_dom"] = []
    summary["no_hits_pmt"] = []
    summary["energy_dom"] = []
    summary["energy_pmt"] = []
    
    # reco = dict()
    # reco["energy"] = []
    # reco["zenith"] = []
    # reco["azimuth"] = []
    
    for event_file_name in filename_list:
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics() #daq() # Get next P frame if it exists
            except:
                print("no daq frame?")
                continue
            # get all pulses
            pulseseriesmap = None
            try:
                if pulse_type == "cleaned":
                    pulseseriesmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, "SRTCleanedInIcePulses")
                    #I3RecoPulseSeriesMapMask?
                #elif pulse_type == "uncleaned":
                #   pulseseriesmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, "PMTResponse")
                else:
                    raise RuntimeError("Unknown pulseseries type specified: %s"%pulse_type)
            except:
                pulseseriesmap = None
            if pulseseriesmap is None:
                print("Broken pulse_series_map - skipping event.")
                continue
            weight = 1.0 #frame["MuonEffectiveArea"]
            nu = frame["MCMuon"]
            nu_energy = nu.energy,
            #print(nu_energy[0])
            nu_zen    = nu.dir.zenith,
            nu_azi    = nu.dir.azimuth,
            nu_x      = nu.pos.x,
            nu_y      = nu.pos.y,
            nu_z      = nu.pos.z,
            dir_x     = nu.dir.x
            dir_y     = nu.dir.y
            dir_z     = nu.dir.z
            # if frame.Has("linefit"):
            #     reco_frame = frame["linefit"]
            #     reco_energy = reco_frame.energy
            #     reco_zenith = reco_frame.dir.zenith
            #     reco_azimuth = reco_frame.dir.azimuth
            # else:
            #     reco_energy = 0
            #     reco_zenith = 0
            #     reco_azimuth = 0

            #MuonType = dataclasses.I3Particle.unknown
            #isTrack = False
            #isCascade = False

            # set track classification for NuMu CC only
            #if frame["I3MCTree"][1].type == dataclasses.I3Particle.MuMinus or frame["I3MCTree"][1].type == dataclasses.I3Particle.MuPlus:
            #    track_length = frame["I3MCTree"][1].length
            #    MuonType = frame["I3MCTree"][1].type
            #    isTrack = True
            #else:
            #    isCascade = True
            #    print("Second particle in MCTree not muon for numu CC? Skipping event...")
            #    continue
            # calculate the event weight
            labels["energy"].append(nu_energy[0])
            labels["azimuth"].append(nu_azi[0]/I3Units.rad)
            labels["zenith"].append(nu_zen[0]/I3Units.rad)
            labels["vtx_x"].append(nu_x[0])
            labels["vtx_y"].append(nu_y[0])
            labels["vtx_z"].append(nu_z[0])
            labels["dir_x"].append(dir_x)
            labels["dir_y"].append(dir_y)
            labels["dir_z"].append(dir_z)

            # reco["energy"].append(reco_energy/I3Units.GeV)
            # reco["zenith"].append(reco_zenith/I3Units.rad)
            # reco["azimuth"].append(reco_azimuth/I3Units.rad)

            dom_index = []
            pmt_index = []
            pulse_time = []
            pulse_charge = []

            for omkey, pulseseries in pulseseriesmap: # Go through each event
                string_num = omkey.string
                om_num = omkey.om
                pmt_num = omkey.pmt

                # if string_num < 1 or string_num > 70:
                #     continue

                # if om_num < 1 or om_num > 20:
                #     #print(string_num,om_num)

                #     # convert string and om into a single index, starting at 1 [not 0 which is special in here and means "no data"]
                #     # OM(1,1) has index 1 - changed to start at 0
                #     continue
                dom_ind = (string_num-1)*80+(om_num-1)
                pmt_ind = (string_num-1)*80+(om_num-1)*24+(pmt_num-1)
                for pulse in pulseseries: # Grab pulse information
                    dom_index.append(dom_ind)
                    pulse_time.append(pulse.time)
                    pulse_charge.append(pulse.charge)
                    pmt_index.append(pmt_ind)

            pulse_time = np.asarray(pulse_time, dtype=np.float64)
            pulse_charge = np.asarray(pulse_charge, dtype=np.float32)
            dom_index = np.asarray(dom_index, dtype=np.uint16)
            pmt_index = np.asarray(pmt_index, dtype=np.uint16)

            # sort the arrays by time (second "feature", index 1)
            sorting = np.argsort(pulse_time)
            pulse_time = pulse_time[sorting]
            pulse_charge = pulse_charge[sorting]
            dom_index = dom_index[sorting]
            pmt_index = pmt_index[sorting]

            # convert absolute times to relative times
            #pulse_time[1:] -= pulse_time[:-1]
            #pulse_time[0] = 0.
            #avg_time = np.mean(pulse_time)
            #pulse_time -= avg_time
            pulse_time = np.asarray(pulse_time, dtype=np.float32)

            features["dom_index"].append(dom_index)
            features["pulse_time"].append(pulse_time)
            features["pulse_charge"].append(pulse_charge)
            features["pmt_index"].append(pmt_index)
            
            no_hits_dom = []
            energy_dom = []
            used = []
            pulse_dom = []
            energy = []
            if len(dom_index) != 0:
                for j in range(len(dom_index)):
                    dom = dom_index[j]
                    if dom not in used:
                        used.append(dom)
                        pulse_dom.append([])
                        energy.append(nu_energy[0])
                    n = used.index(dom)
                    pulse_dom[n].append(pulse_charge[j])
                for k in range(len(used)):
                    no_hits_dom.append(len(pulse_dom[k]))
                    energy_dom.append(energy[k])

            no_hits_pmt = []
            energy_pmt = []
            used = []
            pulse_pmt = []
            energy = []
            if len(pmt_index) != 0:
                for j in range(len(pmt_index)):
                    pmt = pmt_index[j]
                    if pmt not in used:
                        used.append(pmt)
                        pulse_pmt.append([])
                        energy.append(nu_energy[0])
                    n = used.index(pmt)
                    pulse_pmt[n].append(pulse_charge[j])
                for k in range(len(used)):
                    no_hits_pmt.append(len(pulse_pmt[k]))
                    energy_pmt.append(energy[k])


            no_hits_dom = np.asarray(no_hits_dom, dtype=np.float32)
            no_hits_pmt = np.asarray(no_hits_pmt, dtype=np.float32)
            energy_dom = np.asarray(energy_dom, dtype=np.float32)
            energy_pmt = np.asarray(energy_pmt, dtype=np.float32)
            
            summary["no_hits_dom"].append(no_hits_dom)
            summary["no_hits_pmt"].append(no_hits_pmt)
            summary["energy_dom"].append(energy_dom)
            summary["energy_pmt"].append(energy_pmt)

            weights.append(weight)

            del pulseseriesmap

        event_file.close()

    for k in labels.keys():
        labels[k] = np.asarray(labels[k], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return (features, labels, summary, weights )

def write_hdf5_file(filename, features, labels, reco, weights):
    f = h5py.File(output_file, 'w')
    grp_features = f.create_group("features")
    grp_labels   = f.create_group("labels")
    grp_summary  = f.create_group("summary")
    f.create_dataset("weights", data=weights)

    # if reco != None:
    #     grp_reco = f.create_group("reco")

    for k in summary.keys():
        dt = h5py.special_dtype(vlen=summary[k][0].dtype)
        dset = grp_summary.create_dataset(k, (len(summary[k]), ), dtype=dt)

        for i in range(len(summary[k])):
            dset[i] = summary[k][i]

    # for k in summary.keys():
    #     grp_summary.create_dataset(k, data=summary[k])


    for k in labels.keys():
        grp_labels.create_dataset(k, data=labels[k])

    for k in features.keys():
        features[k]
        dt = h5py.special_dtype(vlen=features[k][0].dtype)
        dset = grp_features.create_dataset(k, (len(features[k]), ), dtype=dt)
    
        for i in range(len(features[k])):
            dset[i] = features[k][i]

    # if reco != None:
    #     for k in reco.keys():
    #         grp_reco.create_dataset(k, data=reco[k])

    f.close()   

def strip_i3_ext(filename, keep_path=True):
    path, name = os.path.split(filename)

    while True:
        basename, ext = os.path.splitext(os.path.basename(name))
        if (ext == '') or (ext == ".i3"):
            if keep_path:
                return os.path.join(path, basename)
            else:
                return basename
        name = basename

if '*' in input_files or '?' in input_files:
    input_files = sorted(glob.glob(input_files))

if isinstance(input_files, list):
    for input_file in input_files:
        output_file = "/mnt/home/daummic1/IceCube/summary_hdf5/" + strip_i3_ext(input_file, keep_path=False) + ".hdf5"

        if os.path.isfile(output_file) == True and overwrite == False:
            print("Skipping file -- %s already exists"%output_file)

        else:
            print("Reading {}...".format(input_file))
            features, labels, summary, weights= read_files([input_file])

            # if sum(reco["energy"]) == 0:
            #     reco = None

            if len(weights) > 0 and os.path.isfile(output_file) == False:
                print("Writing {}...".format(output_file))
                write_hdf5_file(output_file, features, labels, summary, weights)
            elif len(weights) > 0 and os.path.isfile(output_file) == True and overwrite == True:
                print("Overwriting {}...".format(output_file))
                write_hdf5_file(output_file, features, labels, summary, weights)
            else:
                print("No output to write, file {} is empty".format(input_file))

elif isinstance(input_files, str):
    input_file = input_files
    output_file = "/mnt/home/daummic1/IceCube/gen2_MC/training2/" + strip_i3_ext(input_file, keep_path=False) + ".hdf5"

    if os.path.isfile(output_file) == True:
        print("Skipping file -- %s already exists"%output_file)

    else:
        print("Reading {}...".format(input_file))
        features, labels, summary, weights = read_files([input_file])

        # if sum(reco["energy"]) == 0:
        #     reco = None

        if len(weights) > 0 and os.path.isfile(output_file) == False:
            print("Writing {}...".format(output_file))
            write_hdf5_file(output_file, features, labels, summary, weights)
        elif len(weights) > 0 and os.path.isfile(output_file) == True and overwrite == True:
            print("Overwriting {}...".format(output_file))
            write_hdf5_file(output_file, features, labels, summary, weights)
        else:
            print("No output to write, file {} is empty".format(input_file))

else:
    print("Unknown data type for input file(s):", type(input_files))
