#!/bin/bash
# #SBATCH --job-name plastic_lug_3D_5000
# #SBATCH --partition cpu
# ##SBATCH --partition gpuA40x4-interactive
# #SBATCH --mem 64gb
# #SBATCH --nodes 1
# ### Testing Hybrid options of Slurm, 4MPI/node, 10SMP/MPI 
# #SBATCH --ntasks-per-node 8
# ##SBATCH --cpus-per-task 8
# #SBATCH --account=bblv-delta-cpu
# #SBATCH --time 18:00:00
# ### GPU options ###
# ##SBATCH --gpus-per-node=1
# ##SBATCH --gpu-bind=none


#ABA_INPUT_FILE=uniaxial.inp
#ABA_INPUT_FILE=thermo_slice_rand_flux_amp.inp


##### Generally no need to edit below this line #####

ABA_EXE=/projects/bbkg/Abaqus/2022/Commands/abq
# ABA_JOB_NAME=${SLURM_JOB_NAME}



#cd ${SLURM_SUBMIT_DIR}
#ln -s Generate_Inputs2 based Abaqus inputs 
InPJobID=$1
cat << EOF > abaqus_v6.env
#mp_host_split=4
scratch="$PWD"
run_mode=INTERACTIVE
#run_mode=BATCH for multiple runs 
license_server_type=FLEXNET
abaquslm_license_file="1715@141.142.193.80"
mp_mode=THREADS
EOF


##${ABA_EXE} job=${ABA_JOB_NAME} input=${ABA_INPUT_FILE} cpus=${SLURM_NTASKS}
#${ABA_EXE} job=${ABA_JOB_NAME} input=${ABA_INPUT_FILE} cpus=2 mp_mode=threads
mkdir inputs.completed
mkdir inputs.failed
#num_jobs=5000
num_jobs=$2
numb=`echo ($num_jobs/3) + 1 |bc`
echo "Number of Batches (of 3) Abaqus runs is $numb for total number of jobs $num_jobs"

touch results.summary
jn=0
#Create a batch loop
for nb in {1..$numb}
do 
# Create loop of 3 runs and wait for them to finish before next set is picked up
# pick 3 inputs at a time 
#for i in $(seq 0 $(($num_jobs-1))); do
# Second loop over nodes 
   for nn in (seq 1 $NNODES); do 
# Third  loop of 3 inputs at a time per node
     for li in (seq 1 3); do 
	     jn= ($nb-1)*3*$nn + $li
	     echo "Abaqus Jobs Index to be launched, $jn" 
	job_name="Job_$jn"
	# submit the run in a separate node based on loop id $li 
	${ABA_EXE} job=${job_name} input=${job_name} cpus=8 double=both & 
	wait
       
	result=`tail -n1 g${job_name}.sta | awk '{print $5}'`
        if [ "$result" eq "SUCCESSFULLY"]; then 
           echo "${job_name} Finished $result" >> results.summary
           mv ${job_name}.inp inputs.completed
	else 
	   echo "${job_name}  Failed "
	   mv ${job_name}.inp inputs.failed
        fi
     done # runs loop
 done # Nodes loop
done # batch loop
rm -fr *.prt *.com
