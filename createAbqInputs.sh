#!/bin/sh -x

#module reset 
#module load anaconda3
source   /projects/cqj/svcscigapgwuser/Abqenv/bin/activate
export username=$AIRAVATA_USERNAME
#export exptname=$EXPERIMENT_NAME
#export useremail=$USER_EMAIL

while getopts r:t:a:n: option
 do
  case $option in
    r ) refin=$OPTARG ;;
    t ) timfl=$OPTARG ;;
    a ) amplf=$OPTARG ;;
    n ) ninps=$OPTARG ;;
     \? ) cat << ENDCAT1
>! Usage: $0  [-r Referencei Input File ]    !<
>!            [-t Time file ]      !<
>!            [-a Amplitudes File ]      !<
>!            [-n Number of runs ]      !<
ENDCAT1
#   exit 1 ;;
  esac
done

echo "User name: $username"
#echo "User email: $useremail"
#echo "Job name: $exptname"
echo "Reference File: $refin"
echo "Time File: $timfl"
echo "Amplitude File: $amplf"
echo "Number of Inputs: $ninps"
mkdir -p /u/svcscigapgwuser/projects/$username
#mkdir -p /u/svcccggwuser/projects/$username
ln -s $PWD /u/svcscigapgwuser/projects/$username/$SLURM_JOBID
python /projects/cqj/svcccggwuser/GENERATE_DATA_5000_REFINED_30K_NODES/create_inp_PlasticLugHard_2022_Gen.py -r $refin -t $timfl -a $amplf -n $ninps
#python /u/svcccggwuser/projects/GENERATE_DATA_5000_REFINED_30K_NODES/create_inp_PlasticLugHard_2022_Gen.py -r $refin -t $timfl -a $amplf -n $ninps


