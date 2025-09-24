import subprocess
import numpy as np
index_failed = []
data_amp_all = np.load('../data_amp_5000.npy')
ystress_all = np.load('ystress_all_new.npy')
peeq_all = np.load('peeq_all_new.npy')
N_train = len(data_amp_all)
for num_sim_i in range(N_train):
	name = '../generated_inps_2/Job_'+ str(num_sim_i) +'.sta'
	pattern = 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'
	try:
           subprocess.check_output(f"grep '{pattern}' {name}", shell=True).decode().strip()

           #print("Job ", num_sim_i, " Converged")
	except:
        	#print("Job ", num_sim_i, " Failed")
              index_failed.append(num_sim_i)
print("index_failed = ", index_failed)
data_amp_reduced = np.delete(data_amp_all, index_failed, axis=0)
ystress_reduced =  np.delete(ystress_all, index_failed, axis=0)
peeq_reduced = np.delete(peeq_all, index_failed, axis=0)


np.save('data_amp_reduced.npy', data_amp_reduced)
np.save('data_stress_reduced.npy', ystress_reduced)
np.save('data_peeq_reduced.npy', peeq_reduced)

print('data_amp_reduced.shape = ', data_amp_reduced.shape)
print('data_stress_reduced.shape = ', ystress_reduced.shape)
print('data_peeq_reduced.shape = ', peeq_reduced.shape)

##data_amp_reduced_check = np.load('data_amp_reduced.npy')
##print('data_amp_reduced_check.shape = ', data_amp_reduced_check.shape)

