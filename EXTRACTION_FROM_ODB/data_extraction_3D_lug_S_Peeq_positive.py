# -*- coding: mbcs -*-
import numpy as np

from abaqus import *
from abaqusConstants import *
import __main__


import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
from odbAccess import *
from types import IntType
from os.path import exists


N_train = 5000
ystress_train_all = []
peeq_train_all = []
amp_all = []
failed = []

for num_sim_i in range(N_train):
    try:
        print >> sys.__stdout__, 'Extracting data from sim_ ' + str(num_sim_i)

        o1 = session.openOdb(name= '../generated_inps_2/Job_'+ str(num_sim_i) +'.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=o1)
        # odb = session.odbs['thermo_slice_2.odb']
        print >> sys.__stdout__, 'after reading assembly'
        # Get the node set containing all nodes in the domain
        print >> sys.__stdout__, o1.rootAssembly.instances.keys()
        instance = o1.rootAssembly.instances['LUG-1']
        print >> sys.__stdout__, 'after reading instance'
        Nofnodes = len(instance.nodes)
        ######node_set = o1.getNodeSet('my_node_set')

        # nodes
        print >> sys.__stdout__, str(Nofnodes)

        #Initialising temporary variables for node coordinates,stress
        N=[]
        NvMS=[]
        #Stress output matrix has values at all nodes from multiple elements
        strval=o1.steps['LugLoad'].frames[-1].fieldOutputs['S'].values
        for j in strval:
           N.append(j.nodeLabel)
           NvMS.append(j.mises)

        #Averaging all element contributions of vM stresses at each node
        Nod, ind=np.unique(N,return_inverse=True,axis=0)
        values = np.array(NvMS)
        indexes = np.array(ind)
        index_set = set(indexes)
        avg_vMS = [np.mean(values[indexes==k]) for k in index_set]
        Ncord = [np.array([instance.nodes[k].coordinates[0], instance.nodes[k].coordinates[1], instance.nodes[k].coordinates[2]]) for k in index_set]
        avg_vMS = np.reshape(avg_vMS,(len(avg_vMS),1))

        ystress_train_all.append(avg_vMS)

        peeq_vals = o1.steps['LugLoad'].frames[-1].fieldOutputs['PEEQ'].values
        N = []
        NvPEEQ = []
        for j in peeq_vals:
            N.append(j.nodeLabel)
            NvPEEQ.append(j.data)

        Nod, ind = np.unique(N, return_inverse=True, axis=0)
        values = np.array(NvPEEQ)
        #### turn all negative nodal ppeq values into zero 
        values[values < 0] = 0.0
        ### cap to small plastic strain only 5% 
        values[values > 0.05] = 0.05
        indexes = np.array(ind)
        index_set = set(indexes)
        avg_PEEQ = [np.mean(values[indexes == k]) for k in index_set]
        avg_PEEQ = np.reshape(avg_PEEQ, (len(avg_PEEQ), 1))

        peeq_train_all.append(avg_PEEQ)


        o1.close()
        '''


        stress_field = o1.steps['ANALYSIS'].frames[-1].fieldOutputs['MISES'].values

        # Initialize lists to store node labels and stress values
        node_labels = []
        stresses = []

        # Iterate over the stress field values
        for stress_val in stress_field:
            node_label = stress_val.nodeLabel
            stress = stress_val.data
            node_labels.append(node_label)
            stresses.append(stress)

        # Print the node labels and corresponding stresses
        for i in range(len(node_labels)):
            print("Node {}: Stress = {}".format(node_labels[i], stresses[i]))
            # print >> sys.__stdout__, str(node_labels[i])
            # print >> sys.__stdout__, str(stresses[i])

        Nod, ind = np.unique(node_labels,return_inverse=True,axis=0)

        stress_values = np.zeros((Nofnodes))
        # values = np.array(stresses)

        indexes = np.array(ind)
        index_set = set(indexes)

        Ncord = np.zeros((Nofnodes,3))
        # print >> sys.__stdout__, str(Ncord.shape)
        # print >> sys.__stdout__, str(stress_values.shape)

        # xy_train_testing = np.zeros((Nofnodes, 2))

        for k,i in enumerate(index_set):
            # print >> sys.__stdout__, str(k)
            # print >> sys.__stdout__, str(i)
            Ncord[i][0] = instance.nodes[k].coordinates[0]
            Ncord[i][1] = instance.nodes[k].coordinates[1]
            Ncord[i][2] = instance.nodes[k].coordinates[2]

            # xy_train_testing[k][0] = instance.nodes[k].coordinates[0]
            # xy_train_testing[k][1] = instance.nodes[k].coordinates[1]

            stress_values[i] = stresses[i]
        # Ncord = [np.array([instance.nodes[k].coordinates[0], instance.nodes[k].coordinates[1], instance.nodes[k].coordinates[2]]) for k in index_set]


        #one training datapoint(rows--number of nodes;1,2 column--nodal coordinate,3 column--vM stress)
        # ystress_train_example=np.concatenate([Ncord,stress_values],axis=1)
        # print >> sys.__stdout__, str(ystress_train_example)


        ystress_train_example = stress_values.T
        # print >> sys.__stdout__, str(ystress_train_example)
        # print >> sys.__stdout__, ystress_train_example.shape
        ystress_train_all.append(ystress_train_example)
        o1.close()
        '''

    except:
        failed.append(num_sim_i)
        print >> sys.__stdout__, 'Extraction failed from sim_ ' + str(num_sim_i)
        pass


#conver ystress_train_all from 3D (#samples, #nodes, 1) into 2D (#samples, #nodes)
ystress_train_all = np.array(ystress_train_all)
ystress_train_all = ystress_train_all.reshape(-1, ystress_train_all.shape[1])

peeq_train_all = np.array(peeq_train_all)
peeq_train_all = peeq_train_all.reshape(-1, peeq_train_all.shape[1])

np.save("ystress_all_new.npy", ystress_train_all)
np.save("peeq_all_new.npy", peeq_train_all)
##np.save('failed_sims.npy', failed)

np.save("xyz_coords.npy",Ncord)
