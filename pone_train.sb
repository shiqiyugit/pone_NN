#!/bin/bash --login

########### SBATCH Lines for Resource Request ##########

#SBATCH --time=23:59:59             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --gres=gpu:v100:1 
#SBATCH --mem=80G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name RNN    # you can give your job a name for easier identification (same as -J)
#SBATCH --output upgrade_RNN_nonstop_log
#SBATCH -A general
#SBATCH --qos normal    
########### Command Lines to Run ##########
dir=/mnt/home/yushiqi2/Analysis/SPACNN/
#singularity exec -B /mnt/home/yushiqi2:/mnt/home/yushiqi2 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python RNN.py --hits 250 --epochs 100 --decay 0.8 --lr 0.001 --dropout 0.1 --log_energy 0 --standardize 0 --checkpoints 0 --weights 0 --data_type 'level2_E0_500_CC_vertex_start_end_DC'
#source activate tfgpu-brandon

export PATH="/mnt/home/priesbr1/anaconda3/bin:/mnt/home/priesbr1/anaconda3/condabin:/opt/software/powertools/bin:/opt/software/MATLAB/2018a:/opt/software/MATLAB/2018a/bin:/opt/software/Java/1.8.0_152:/opt/software/Java/1.8.0_152/bin:/opt/software/Python/3.6.4-foss-2018a/bin:/opt/software/SQLite/3.21.0-GCCcore-6.4.0/bin:/opt/software/Tcl/8.6.8-GCCcore-6.4.0/bin:/opt/software/libreadline/7.0-GCCcore-6.4.0/bin:/opt/software/ncurses/6.0-GCCcore-6.4.0/bin:/opt/software/CMake/3.11.1-GCCcore-6.4.0/bin:/opt/software/bzip2/1.0.6-GCCcore-6.4.0/bin:/opt/software/FFTW/3.3.7-gompi-2018a/bin:/opt/software/OpenBLAS/0.2.20-GCC-6.4.0-2.28/bin:/opt/software/imkl/2018.1.163-gompi-2018a/mkl/bin:/opt/software/imkl/2018.1.163-gompi-2018a/bin:/opt/software/OpenMPI/2.1.2-GCC-6.4.0-2.28/bin:/opt/software/binutils/2.28-GCCcore-6.4.0/bin:/opt/software/GCCcore/6.4.0/bin:/usr/lib64/qt-3.3/bin:/opt/software/core/lua/lua/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/local/hpcc/bin:/usr/lpp/mmfs/bin:/opt/ibutils/bin:/opt/puppetlabs/bin"

cd $dir
#/mnt/research/IceCube/willey/Upgrade_RNN/

source activate /mnt/home/priesbr1/anaconda3/envs/tfgpu-brandon
#singularity exec -B /mnt/home/yushiqi2:/mnt/home/yushiqi2 -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif 

#python $dir/RNN.py --hits 250 --epochs 100  --beta_1 0.8 --lr 0.0001 --dropout 0.1 --log_energy 0 --standardize 0 --weights 0 --data_type 'level2_E0_500_CC_vertex_start_end_DC_loss_adamax_abs_error' -c 1


python $dir/pone_RNN.py --hits 300 --epochs 1000 --beta_1 0.8 --lr 0.002 --dropout 0.2 --log_energy 0 --standardize 0 --weights 0 --data_type 'cleaned_linefit_300_dxyz_10DOM_10TeV' -c 1 -f "pone_cleaned_pulses_linefit_10minDOM_min10000GeV.hdf5" 
#pone_cleaned_pulses_linefit_20minDOM.hdf5"

#python $dir/RNN.py --hits 250 --epochs 5 --beta_1 0.8 --lr 0.0001 --dropout 0 --log_energy 0 --standardize 0 --weights 0 --data_type 'level2_E0_500_CC_vertex_start_end_DC_loss_adam' -c 1

exit $?
