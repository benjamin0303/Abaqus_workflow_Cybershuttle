#!/bin/bash

#SBATCH --job-name aba_extraction_5000_S_PEEQ_with_fiter
#SBATCH --output aba_extraction_5000_S_PEEQ_with_filter.out
#SBATCH --error aba_extraction_5000_S_PEEQ_with_filter.err
#SBATCH --partition cpu
#SBATCH --mem 64gb
#SBATCH --nodes 1
### Testing Hybrid options of Slurm, 4MPI/node, 10SMP/MPI
#SBATCH --ntasks-per-node 1
##SBATCH --cpus-per-task 8
#SBATCH --account=bblv-delta-cpu
#SBATCH --time 48:00:00
### GPU options ###
##SBATCH --gpus-per-node=1
##SBATCH --gpu-bind=none


ABA_PYTHON_FILE=data_extraction_3D_lug_S_Peeq_positive.py


##### Generally no need to edit below this line #####

ABA_EXE=/projects/bbkg/Abaqus/2022/Commands/abq
# ABA_JOB_NAME=${SLURM_JOB_NAME}



cd ${SLURM_SUBMIT_DIR}


cat << EOF > abaqus_v6.env
#mp_host_split=4
scratch="$PWD"
run_mode=INTERACTIVE
license_server_type=FLEXNET
abaquslm_license_file="1715@141.142.193.80"
mp_mode=THREADS
EOF

${ABA_EXE} cae noGUI=${ABA_PYTHON_FILE}

#filter our data that did not converge by grep in sta files

module load anaconda3_gpu/22.10.0 

srun python3 filter_out_failed_data.py
