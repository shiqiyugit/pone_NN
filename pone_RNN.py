import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
import h5py
import scipy.stats
import glob
import os
import sys
import math
import argparse

from Generators import DataGenerator, SplitGenerator
from Attention import AttentionWithContext
from Plots import plot_uncertainty, plot_uncertainty_2d, plot_2dhist, plot_2dhist_contours, plot_1dhist, plot_error, plot_error_contours, plot_loss, plot_error_vs_reco, plot_inputs, plot_outputs

import keras
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from keras import backend as K, initializers, regularizers, constraints
from keras.layers import Layer
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Embedding, BatchNormalization
from keras.optimizers import Adam

from keras.layers import Lambda, Flatten, Reshape, CuDNNLSTM, LSTM, Bidirectional, Activation, Dropout
from keras.layers import Conv1D, SpatialDropout1D, MaxPooling1D

np.set_printoptions(threshold=sys.maxsize)
VOCAB_SIZE= 70*20

def zenith_loss(y_true, y_pred):
    return keras.losses.mean_absolute_error(y_true[:,1],y_pred[:,1])+keras.losses.mean_absolute_error(y_true[:,2],y_pred[:,2])+keras.losses.mean_absolute_error(y_true[:,3],y_pred[:,3])

def direction_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true[:,0],y_pred[:,0])+keras.losses.mean_squared_error(y_true[:,1],y_pred[:,1])+keras.losses.mean_squared_error(y_true[:,2],y_pred[:,2])

def angles2vector(angles):
    """
    Convert 2D angle (yaw and pitch) to 3D unit vector
    :param angles: list of 2D angles
    :return: computed 3D vectors
    """
    x = (-1.0) * K.sin(angles[:, 0]) * K.cos(angles[:, 1])
    y = (-1.0) * K.sin(angles[:, 1])
    z = (-1.0) * K.cos(angles[:, 0]) * K.cos(angles[:, 1])
    vec = K.transpose(K.concatenate([[x], [y], [z]], axis=0))
    return vec 

def angle_error(gt, pred):
    """
    Average angular error computed by cosine difference
    :param gt: list of ground truth label
    :param pred: list of predicted label
    :return: Average angular error in radians
    """
    vec_gt = gt #angles2vector(gt)
    vec_pred = pred #angles2vector(pred)

    x = K.np.multiply(vec_gt[:, 0], vec_pred[:, 0])
    y = K.np.multiply(vec_gt[:, 1], vec_pred[:, 1])
    z = K.np.multiply(vec_gt[:, 2], vec_pred[:, 2])

    dif = K.np.sum([x, y, z], axis=0) / (tf.norm(vec_gt, axis=1) * tf.norm(vec_pred, axis=1))

    clipped_dif = K.clip(dif, np.float(-1.0), np.float(1.0))
    loss = (tf.acos(clipped_dif) * 180) / np.pi
    return K.mean(loss, axis=-1) 

def customLoss(y_true, y_pred):
    d_loss = angle_error(y_true, y_pred) #direction_loss(y_true, y_pred)
#    d_loss = direction_loss(y_true, y_pred)
    return d_loss

def to_xyz(zenith, azimuth):
    theta = np.pi-zenith
    phi = azimuth-np.pi
    rho = np.sin(theta)
    return rho*np.cos(phi), rho*np.sin(phi), np.cos(theta)
    
def angular_error(y_pred, y_true):
    doc=[]
    for i in range(y_pred.shape[0]):
        doc.append(np.dot(y_pred[i], y_true[i])/(np.linalg.norm(y_pred[i])*np.linalg.norm(y_true[i])))
    ret = np.full(len(y_pred), np.inf)
    ret = np.degrees(np.arccos(doc))
    return ret

def to_zenazi(x,y,z):
    from astropy.coordinates import SkyCoord, cartesian_to_spherical
#    c = SkyCoord(x=x, y=y, z=z, representation_type='cartesian')
    rho, phi, theta = cartesian_to_spherical(x, y, z)
#c.cartesian_to_sphericaltransform_to('icrs')
    zenith = np.pi - theta.value
    azimuth = phi.value + np.pi
    return zenith, azimuth
"""
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.zeros(len(r))
        
    normal_bins = (r>0.0) & (np.abs(np.asarray(z)/r)<=1.0)
    theta[normal_bins] = np.arccos(z[normal_bins]/r[normal_bins])
    theta[np.logical_not(normal_bins) & (np.asarray(z) < 0.0)] = np.pi
    theta[theta<0.0] += 2.0*np.pi
    
    phi = np.zeros(len(r))
    mask=(np.asarray(x)!=0.0) & (np.asarray(y)!=0.0)
    phi[mask ] = np.arctan2(y[mask],x[mask])
    phi[phi < 0.0] += 2.0*np.pi

    zenith = np.pi - theta
    azimuth = phi + np.pi
   
    zenith[zenith > np.pi] -= np.pi-(zenith[zenith > np.pi]-np.pi)
    azimuth -= (azimuth/(2.0*np.pi)).astype(np.int).astype(np.float) * 2.0*np.pi
    
    return zenith, azimuth
"""
def forward_generators(gen_train, gen_val, last_checkpoint_epoch):

    print("fast-forwarding generators...")
    initial_epoch = 0
    while initial_epoch < last_checkpoint_epoch:
        # request at least one item, just to make sure
        print("  forwarding one epoch...")

        dummy = gen_train[0]
        dummy = gen_val[0]
        del dummy

        gen_train.on_epoch_end()
        gen_val.on_epoch_end()

        initial_epoch += 1

    return gen_train, gen_val 
    
def main(config=1):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--hits",type=int,default=150, dest="hits", help="number of dom hits used for training")
    parser.add_argument("-e", "--epochs",type=int,default=8, dest="epochs", help="number of training epochs")
    parser.add_argument("-b", "--beta_1",type=float,default=0.8, dest="beta_1", help="learning rate decay parameter")
    parser.add_argument("-r", "--lr", type=float,default=0.0001, dest="lr", help="learning rate")
    parser.add_argument("-o", "--dropout", type=float,default=0., dest="dropout", help="change network dropout for each layer")
    parser.add_argument("-l", "--log_energy", type=int,default=0, dest="log_energy", help="use log energy rather than absolute for training")
    parser.add_argument("-f", "--file", type=str, default="pone_cleaned_pulses_linefit_10minDOM.hdf5", dest="file_name", help="file to use for training")
    parser.add_argument("-p", "--path", type=str, default="/mnt/scratch/yushiqi2/", dest="path", help="path to input file")
    parser.add_argument("-u", "--output", type=str, default="/mnt/scratch/yushiqi2/pone_RNN/", dest="output", help="output folder destination")
    parser.add_argument("-s", "--standardize", type=int,default=0, dest="standardize", help="perform data standardization")
    parser.add_argument("-c", "--checkpoints", type=int,default=0, dest="checkpoints", help="use training checkpoints from previous run")
    parser.add_argument("-w", "--weights", type=int,default=1, dest="weights", help="use sample weights for training")
    parser.add_argument("-t", "--data_type", type=str, default=None, help="name")
    args = parser.parse_args()

    no_hits = args.hits
    no_epochs = args.epochs
    beta_1 = args.beta_1
    learning_rate = args.lr
    dropout = args.dropout
    use_log_energy = bool(args.log_energy)
    ff_name = args.path + args.file_name
    use_standardization = bool(args.standardize)
    use_checkpoints = bool(args.checkpoints)
    use_weights = bool(args.weights)

    ff = h5py.File(ff_name, 'r')
    global gen_filename
    global save_folder_name
    gen_filename = "run_"+args.data_type+"_angular_lr-4"+'_'+str(int(no_hits))+"hits" #+str(int(np.log10(learning_rate)))+'_'+str(int(no_hits))+"hits"

    save_folder_name = args.output + gen_filename + '/'
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
    print("Saving to:", save_folder_name)

    reco = False
    if "reco" in ff.keys():
        reco = True
    network_labels = ["dir_x", "dir_y", "dir_z"]#, "energy"]#, "zenith", "azimuth", "energy"]
    if use_standardization:
        normalization = normalize(ff, network_labels, use_log_energy)
    else:
        normalization = []
    gen = DataGenerator(ff, labels=network_labels, maxlen=no_hits, use_log_energy=use_log_energy,use_weights=use_weights,normal=normalization)
    gen_train = SplitGenerator(gen, fraction=0.70, offset=0.00)
    gen_val = SplitGenerator(gen, fraction=0.3, offset=0.70)
#    test_gen = DataGenerator(ff, labels=network_labels, maxlen=no_hits, use_log_energy=use_log_energy,use_weights=use_weights,normal=normalization)

    test_labels = ["dir_x", "dir_y", "dir_z", "energy"]
    test_gen = DataGenerator(ff, labels=test_labels, maxlen=no_hits, use_log_energy=use_log_energy,use_weights=use_weights,normal=normalization)
    gen_test = SplitGenerator(test_gen, fraction=0.4, offset=0.60)
    vocab_size = VOCAB_SIZE
    time_samples = no_hits

    # Instantiate the base model (or "template" model).
    # We recommend doing this with under a CPU device scope,
    # so that the model's weights are hosted on CPU memory.
    # Otherwise they may end up hosted on a GPU, which would
    # complicate weight sharing.
    input_data = Input(shape=(time_samples,3), name="input_data") # variable length
    input_dom_index = Lambda( lambda x: x[:,:,0], name="input_dom_index" )(input_data) # slice out the dom index
    input_rel_time  = Reshape( (-1,1), name="reshaped_rel_time" )(Lambda( lambda x: x[:,:,1], name="input_rel_time" )(input_data))  # slice out the relative time
    input_charge    = Reshape( (-1,1), name="reshaped_charge" )(Lambda( lambda x: x[:,:,2], name="input_charge" )(input_data))    # slice out the charge

    
    embedding_dom_index = Embedding(input_dim=vocab_size,
                                    output_dim=3,
                                    input_length=time_samples,
                                    # mask_zero=True,
                                    name="embedding_dom_index")(input_dom_index)
    x = Concatenate(axis=-1, name="concatenated_features")([embedding_dom_index, input_rel_time, input_charge])
    
    x = CuDNNLSTM(256, return_sequences=True, name="lstm1")(x)
    x = Dropout(dropout)(x)
    x = CuDNNLSTM(256, return_sequences=True, name="lstm2")(x)
    x = Dropout(dropout)(x)
    x = CuDNNLSTM(256, return_sequences=True, name="lstm3")(x)
    x = Dropout(dropout)(x)
    x = CuDNNLSTM(256, return_sequences=True, name="lstm4")(x)
    x = Dropout(dropout)(x)
    x = AttentionWithContext(name="attention")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout)(x)
    dense_regression = Dense(64, activation="tanh")(x)
    dense_regression = Dropout(dropout)(dense_regression)
    #dense_classification = Dense(64, activation="tanh")(x)
    #dense_classification = Dropout(dropout)(dense_classification)
    
    outputs = Dense( 3, activation="linear",   name="dense_dxdydz")(dense_regression) # range   -1..1
    #dense_tc = Dense(2, activation="sigmoid", name="dense_tc")(dense_classification)
#    outputs = Concatenate(axis=-1, name="output")([dense_energy, dense_dxdydz]) #dense_energy_sig, dense_dxdydz, dense_dxdydz_sig])#dense_tc])
#    outputs = Dense( 3, activation="linear",   name="output")(dense_regression)
#Concatenate(axis=-1, name="output")(dense_dxdydz)
    
    model = Model(inputs=input_data, outputs=outputs)

    opt = keras.optimizers.Adamax(lr=learning_rate,beta_1=beta_1, clipvalue=0.1)#keras.optimizers.SGD(lr=0.01,momentum=0.8)
#    opt = keras.optimizers.SGD(lr=learning_rate,momentum=beta_1)

    model.compile(optimizer=opt, loss=customLoss) #, metrics=[direction_loss])
    
    model.summary()

    # get all files
    checkpoint_files = glob.glob("%sweights.?????.hdf5"%save_folder_name)
    checkpoint_files.sort()
 
    if len(checkpoint_files) == 0:
        print("no checkpoints available, starting from scratch.")
        initial_epoch = 0
    elif not use_checkpoints:
        print("checkpoints not used, starting from scratch.")
        initial_epoch = 0
    else:
        indices = []
        for i in range(len(checkpoint_files)):
            # strip the path
            _, filename = os.path.split(checkpoint_files[i])
            if int(filename[8:8+5]) <= no_epochs:
                print(filename)
                indices.append( int(filename[8:8+5]) )
    
        indices = np.array(indices)
        sorting = np.argsort(indices)
        last_checkpoint = checkpoint_files[ sorting[-1] ]
        last_checkpoint_epoch = indices[ sorting[-1] ]
        initial_epoch = last_checkpoint_epoch
 
        print("Loading epoch {} from checkpoint file {}".format( last_checkpoint_epoch, last_checkpoint ))
        model.load_weights(last_checkpoint)
    
        gen_train, gen_val = forward_generators(gen_train, gen_val, last_checkpoint_epoch)

    if initial_epoch == no_epochs:
        train = False
    else:
        train = True
 
    print("Initial epoch index is {}".format(initial_epoch))
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "%sweights.{epoch:05d}.hdf5"%save_folder_name, 
        monitor="val_loss", 
        save_weights_only=True)

    csv_logger = keras.callbacks.CSVLogger(save_folder_name+'/training.log', separator=',', append=True)
    def schedule_function(epoch, lr=learning_rate, lr_drop=500):
        lr = learning_rate
        print(lr * math.exp(-0.1 * int(epoch/lr_drop)))
        return lr * math.exp(-0.1 * int(epoch/lr_drop))
    
    lr_schedule = keras.callbacks.LearningRateScheduler(schedule_function)
 
    if train:
        print("Training model")
        network_history = model.fit_generator(
            generator=gen_train,
            steps_per_epoch=len(gen_train),
            validation_data=gen_val,
            validation_steps=len(gen_val),
            epochs=no_epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            shuffle=True,
            workers=1,
            use_multiprocessing=False,
            callbacks=[model_checkpoint, csv_logger, lr_schedule])
    
        weightfile_name = save_folder_name+"weightfile.hdf5"
        model.save_weights(weightfile_name)
        model.load_weights(weightfile_name)
    
    labels_raw = None
    labels_predicted_raw = None
    if reco: labels_reco = None
    weights_raw = None
   
    if train: 
        test_metrics = model.evaluate_generator(gen_test)
        train_metrics = model.evaluate_generator(gen_train)
        val_metrics = model.evaluate_generator(gen_val)

    print("Testing model")

    for i in range(len(gen_test)-1):
        batch_features, batch_labels, batch_weights = gen_test[i]
        if reco: batch_reco = gen_test.get_reco(i)
        batch_labels_predicted = model.predict(batch_features)

        if labels_raw is None:
            labels_raw = batch_labels
            labels_predicted_raw = batch_labels_predicted
            if reco: labels_reco = batch_reco
            weights_raw = batch_weights
        else:
            labels_raw           = np.append(labels_raw,           batch_labels,           axis=0)
            labels_predicted_raw = np.append(labels_predicted_raw, batch_labels_predicted, axis=0)
            if reco: labels_reco = np.append(labels_reco, batch_reco, axis=-1)
            weights_raw          = np.append(weights_raw,          batch_weights,          axis=0)
        del batch_labels_predicted
        del batch_features
        del batch_labels
        if reco: del batch_reco
        del batch_weights
 
    labels_predicted = labels_predicted_raw #gen.untransform_labels(labels_predicted_raw)
    labels = labels_raw #gen.untransform_labels(labels_raw)
    weights = weights_raw
#    energy_true = labels[:,0]
#    energy_sigma = labels_predicted[:,1]
    dx_predicted = labels_predicted[:,0]
    dx_true = labels[:,0]
#    dx_sigma = labels_predicted[:,5]
    dy_predicted = labels_predicted[:,1]
    dy_true = labels[:,1]
#    dy_sigma = labels_predicted[:,6]
    dz_predicted = labels_predicted[:,2]
    dz_true = labels[:,2]
    energy_true = labels[:,3]
    print(energy_true.shape, dz_true.shape)
#    dz_sigma = labels_predicted[:,7]
    vec_true = np.column_stack((dx_true, dy_true, dz_true))
    vec_pred = np.column_stack((dx_predicted, dy_predicted, dz_predicted))
    angular_err_pred = angular_error(vec_pred, vec_true)
    zenith_reco, azimuth_reco, angular_err_reco = None, None, None
    if reco:
        # order in generator: energy, azimuth, zenith in unit of radians
        azimuth_reco = np.degrees(labels_reco[1])
        zenith_reco = np.degrees(labels_reco[2])
       
        dx_reco, dy_reco, dz_reco = to_xyz(labels_reco[2], labels_reco[1])
        vec_reco = np.column_stack((dx_reco, dy_reco, dz_reco))
        angular_err_reco = angular_error(vec_reco, vec_true)
        print(angular_err_reco.shape, vec_reco.shape)
#         zenith_reco, azimuth_reco = np.degrees(to_zenazi(dx_reco, dy_reco, dz_reco))

    if train:
        plot_loss(network_history.history, test_metrics, "loss", "Loss", no_epochs, gen_filename=save_folder_name, unc=False)

 
    #isTrack_predicted = labels_predicted[:,4]
    #isCascade_predicted = labels_predicted[:,5]
    #isTrack_true = labels[:,4]
    #isCascade_true = labels[:,5]
    
    #isTrack_predicted = [isTrack_predicted > isCascade_predicted]
    #isCascade_predicted = [isCascade_predicted > isTrack_predicted]
    
    #trueTracks = np.sum(np.logical_and(isTrack_true, isTrack_predicted))
    #falseTracks = np.sum(np.logical_and(np.logical_not(isTrack_true), isTrack_predicted))
    #trueCascades = np.sum(np.logical_and(isCascade_true, isCascade_predicted))
    #falseCascades = np.sum(np.logical_and(np.logical_not(isCascade_true), isCascade_predicted))
    
    #fig, ax = plt.subplots()
    #bars1 = ax.bar(np.arange(2), [trueTracks, trueCascades], 0.25, color="SkyBlue")
    #bars2 = ax.bar(np.arange(2)+0.5*np.ones(2), [falseTracks, falseCascades], 0.25, color="IndianRed")
    #ax.set_title("Track vs. Cascade classification results")
    #ax.set_xticks(np.arange(4)/2)
    #ax.set_xticklabels(("True Tracks", "False Tracks", "True Cascades", "False Cascades"))
    #imgname = save_folder_name+"class.png"
    #plt.savefig(imgname)
    
    zenith_predicted, azimuth_predicted = np.degrees(to_zenazi(dx_predicted, dy_predicted, dz_predicted))
    zenith_true, azimuth_true = np.degrees(to_zenazi(dx_true, dy_true, dz_true))

    r_predicted = np.sqrt(dx_predicted**2+dy_predicted**2+dz_predicted**2)
#    r_sigma = np.sqrt(np.divide((dx_predicted*dx_sigma)**2+(dy_predicted*dy_sigma)**2+(dz_predicted*dz_sigma)**2,r_predicted**2))
#    zenith_sigma = np.degrees(np.sqrt(np.divide((dz_predicted*r_sigma)**2+(r_predicted*dz_sigma)**2,r_predicted**2*(r_predicted**2-dz_predicted**2))))
#    azimuth_sigma = np.degrees(np.sqrt(np.divide((dx_sigma*dy_predicted)**2+(dy_sigma*dx_predicted)**2,(dx_predicted**2+dy_predicted**2)**2)))

    #Make plots
    if reco:
        plot_error_vs_reco(azimuth_true, azimuth_predicted, azimuth_reco, 0, 360, "Azimuth [degrees]", gen_filename=save_folder_name)
        plot_error_vs_reco(zenith_true, zenith_predicted, zenith_reco, 0, 180, "Zenith [degrees]", gen_filename=save_folder_name)
        plot_error_vs_reco(energy_true, angular_err_pred, angular_err_reco, np.nanmin(energy_true), np.nanmax(energy_true), "Angular [deg]","Energy [GeV]", x=energy_true, gen_filename=save_folder_name)
        print("std of pred, linefit: ", np.nanmedian(angular_err_pred), np.nanmedian(angular_err_reco))
        plot_1dhist(angular_err_pred, angular_err_reco, np.nanmin(angular_err_pred), np.nanmax(angular_err_pred), "angular error [deg]", None, gen_filename=save_folder_name, l1="Predicted", l2="Linefit")
        quit()
    else:
        plot_error(azimuth_true, azimuth_predicted, 0, 360, "Azimuth [degrees]", gen_filename=save_folder_name)
        plot_error(zenith_true, zenith_predicted, 0, 180, "Zenith [degrees]", gen_filename=save_folder_name)

    plot_2dhist(dx_true, dx_predicted, -1.0, 1.0, "dx [m]", weights, gen_filename=save_folder_name)
    plot_2dhist(dy_true, dy_predicted, -1.0, 1.0, "dy [m]", weights, gen_filename=save_folder_name)
    plot_2dhist(dz_true, dz_predicted, -1.0, 1.0, "dz [m]", weights, gen_filename=save_folder_name)
    plot_2dhist(azimuth_true, azimuth_predicted, 0, 360, "Azimuth [degrees]", weights, gen_filename=save_folder_name)
    plot_2dhist(zenith_true, zenith_predicted, 0, 180, "Zenith [degrees]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dx_true, dx_predicted, -1.0, 1.0, "dx [m]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dy_true, dy_predicted, -1.0, 1.0, "dy [m]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dz_true, dz_predicted, -1.0, 1.0, "dz [m]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(azimuth_true, azimuth_predicted, 0, 360, "Azimuth [degrees]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(zenith_true, zenith_predicted, 0, 180, "Zenith [degrees]", weights, gen_filename=save_folder_name)
    plot_2dhist_contours(np.cos(zenith_true*np.pi/180), np.cos(zenith_predicted*np.pi/180), -1.0, 1.0, "Cos(Zenith)", weights, gen_filename=save_folder_name)
    plot_1dhist(dx_true, dx_predicted, -1.0, 1.0, "dx [m]", weights, gen_filename=save_folder_name)
    plot_1dhist(dy_true, dy_predicted, -1.0, 1.0, "dy [m]", weights, gen_filename=save_folder_name)
    plot_1dhist(dz_true, dz_predicted, -1.0, 1.0, "dz [m]", weights, gen_filename=save_folder_name)
    plot_1dhist(azimuth_true, azimuth_predicted, 0, 360, "Azimuth [degrees]", weights, gen_filename=save_folder_name)
    plot_1dhist(zenith_true, zenith_predicted, 0, 180, "Zenith [degrees]", weights, gen_filename=save_folder_name)
    plot_error(dx_true, dx_predicted, -1.0, 1.0, "dx [m]", gen_filename=save_folder_name)
    plot_error(dy_true, dy_predicted, -1.0, 1.0, "dy [m]", gen_filename=save_folder_name)
    plot_error(dz_true, dz_predicted, -1.0, 1.0, "dz [m]", gen_filename=save_folder_name)
    plot_error_contours(np.cos(zenith_true*np.pi/180), np.cos(zenith_predicted*np.pi/180), -1.0, 1.0, "Cos(Zenith)", gen_filename=save_folder_name)
#    plot_uncertainty(energy_true, energy_predicted, energy_sigma, "Energy [GeV]", weights, gen_filename=save_folder_name)
#    plot_uncertainty(dx_true, dx_predicted, dx_sigma, "dx [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty(dy_true, dy_predicted, dy_sigma, "dy [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty(dz_true, dz_predicted, dz_sigma, "dz [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty(azimuth_true, azimuth_predicted, azimuth_sigma, "Azimuth [degrees]", weights, gen_filename=save_folder_name)
#    plot_uncertainty(zenith_true, zenith_predicted, zenith_sigma, "Zenith [degrees]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(energy_true, energy_predicted, energy_sigma, min(energy_true), max(energy_true), "Energy [GeV]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(dx_true, dx_predicted, dx_sigma, -1.0, 1.0, "dx [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(dy_true, dy_predicted, dy_sigma, -1.0, 1.0, "dy [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(dz_true, dz_predicted, dz_sigma, -1.0, 1.0, "dz [m]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(azimuth_true, azimuth_predicted, azimuth_sigma, 0, 360, "Azimuth [degrees]", weights, gen_filename=save_folder_name)
#    plot_uncertainty_2d(zenith_true, zenith_predicted, zenith_sigma, 0, 180, "Zenith [degrees]", weights, gen_filename=save_folder_name)

    #output results
    print("DIAGNOSTICS")
    if reco:
        zen_RNN_err = np.absolute(zenith_true[zenith_reco > 0] - zenith_predicted[zenith_reco > 0])
        zen_PL_err = np.absolute(zenith_true[zenith_reco > 0] - zenith_reco[zenith_reco > 0])
        azi_RNN_err = np.absolute(azimuth_true[azimuth_reco > 0] - azimuth_predicted[azimuth_reco > 0])
        azi_PL_err = np.absolute(azimuth_true[azimuth_reco > 0] - azimuth_reco[azimuth_reco > 0])
        azi_PL_err = np.array([azi_PL_err[i] if (azi_PL_err[i] < 180) else (360-azi_PL_err[i]) for i in range(len(azi_PL_err))])
        azi_PL_err = np.array([azi_PL_err[i] if (azi_PL_err[i] > -180) else (360+azi_PL_err[i]) for i in range(len(azi_PL_err))])
        avg_zen_PL_err = np.mean(zen_PL_err)
        avg_azi_PL_err = np.mean(azi_PL_err)
        std_zen_PL_err = np.std(zen_PL_err)
        std_azi_PL_err = np.std(azi_PL_err)
        print("PegLeg")
        print("Azimuth: average absolute error = "+str(avg_azi_PL_err)+", sigma = "+str(std_azi_PL_err))
        print("Zenith: average absolute error = "+str(avg_zen_PL_err)+", sigma = "+str(std_zen_PL_err))
    else:
        zen_RNN_err = np.absolute(zenith_true - zenith_predicted)
        azi_RNN_err = np.absolute(azimuth_true - azimuth_predicted)
    azi_RNN_err = np.array([azi_RNN_err[i] if (azi_RNN_err[i] < 180) else (360-azi_RNN_err[i]) for i in range(len(azi_RNN_err))])
    azi_RNN_err = np.array([azi_RNN_err[i] if (azi_RNN_err[i] > -180) else (360+azi_RNN_err[i]) for i in range(len(azi_RNN_err))])
    avg_zen_RNN_err = np.mean(zen_RNN_err)
    avg_azi_RNN_err = np.mean(azi_RNN_err)
    std_zen_RNN_err = np.std(zen_RNN_err)
    std_azi_RNN_err = np.std(azi_RNN_err)
    print("RNN")
    print("Azimuth: average absolute error = "+str(avg_azi_RNN_err)+", sigma = "+str(std_azi_RNN_err))
    print("Zenith: average absolute error = "+str(avg_zen_RNN_err)+", sigma = "+str(std_zen_RNN_err))

    return 0#network_history.history['val_loss']

if __name__ == "__main__":
    main()
