import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from IPython.display import display
import three_desc_model as exponential_new
from cutoffs import Polynomial
GPy.plotting.change_plotting_library('matplotlib')

# CLASS FOR TRAINING
class GP_train:
    def __init__(self):
        return
    def train_model(self):
        # LOAD DATA
        dataset_full = pd.read_csv("all_dataset.csv", index_col=0)
        symbol="Siemens"
        k=0
        count=0
        dataset_full["IDnum"]=pd.Series()
        for i in range(len(dataset_full)):
            dataset_full.at[i,"IDnum"]=k
            count = count +1
            if (count==dataset_full["Num_tot_turb"].iloc[i]):
                k=k+1
                count=0
        dataset_full["Num_tot_turb"]=dataset_full["Num_tot_turb"].astype(int)
        dataset_full["Turb_num"]=dataset_full["Turb_num"].astype(int)
        dataset_full["IDnum"]=dataset_full["IDnum"].astype(int)
        numsims=dataset_full["IDnum"].iloc[-1]+1
        turb = [symbol]

        # INITIALISE NEIGHBOUR LIST
        nl=exponential_new.NeighborlistCalculator(cutoff=4001,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)

        # INITIALISE FINGPR
        Gs = {"Siemens": [{"type":"G2", "turbine":"Siemens","eta":4.25387599, "offset": 1.0151402},
                          {"type":"G4", "elements":"Siemens","eta":2.56450515, "gamma":8.04475192, "zeta": 2.5356155},
                          {"type":"G6", "elements":"Siemens","eta":2.33043463, "gamma": 0.50753377, "zeta": 0.93372721}
                         ]}
        finpr=exponential_new.FingerprintCalculator(cutoff=4001,Gs=Gs,Rct=3000,delta_R=100,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)

        # CALCULATE NL AND FINGPR
        # outputs: fingerprints and reference velocities
        count=0
        dataset_fp=np.empty(shape=(0, 3))
        dataset_ws=np.empty(shape=(0, 1))
        dataset_pos=np.empty(shape=(0, 2))
        for i in range(numsims):
            numturb=dataset_full["Num_tot_turb"].iloc[count]
            position = np.empty((numturb,2))
            ws = np.empty((numturb,1))
            fp = np.empty((numturb,3))
            for k in range(numturb):
                position[k,0]=dataset_full.at[count,"X_coord"]
                position[k,1]=dataset_full.at[count,"Y_coord"]
                ws[k,0]=dataset_full.at[count,"Ref_wind"]
                count = count+ 1
            neigh=nl.calculate(turb*numturb,position)
            fingerprints=finpr.calculate(turb*numturb,position,neigh,symbol)
            fingerprints=np.array(fingerprints)
            dataset_fp=np.append(dataset_fp,fingerprints,axis=0)
            dataset_ws=np.append(dataset_ws,ws,axis=0)
            dataset_pos=np.append(dataset_pos,position,axis=0)
        dataset=np.concatenate((dataset_fp, dataset_ws),axis=1)
        dataset = pd.DataFrame(dataset, columns = ['Fingerprint(G2)','Fingerprint(G4)','Fingerprint(G6)','Ref_Wind_Speed'])
        X=dataset[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)"]].to_numpy()
        Y=dataset[["Ref_Wind_Speed"]].to_numpy()
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)
        Xtrain=train_dataset[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)"]].to_numpy()
        Ytrain=train_dataset[["Ref_Wind_Speed"]].to_numpy()
        Xtest=test_dataset[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)"]].to_numpy()
        Ytest=test_dataset[["Ref_Wind_Speed"]].to_numpy()

        # DEFINE KERNEL
        ker = GPy.kern.RBF(3,lengthscale=0.1)
        
        # CREATE GP MODEL
        m = GPy.models.GPRegression(Xtrain,Ytrain,ker)
        
        # OPTIMISE
        m.optimize(messages=True,max_f_eval = 1000)
        
        return m


# CLASS FOR PREDICTING
class GP_predict:
    def __init__(self, model):
        self.model = model
        return
    
    def predict(self, pos_array, num_turbs_predict):
        # DECLARE TURBINE TYPE
        symbol="Siemens"
        turb = [symbol]

        # INITIALISE NEIGHBOUR LIST
        nl=exponential_new.NeighborlistCalculator(cutoff=4001,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)

        # INITIALISE FINGPR
        Gs = {"Siemens": [{"type":"G2", "turbine":"Siemens","eta":4.25387599, "offset": 1.0151402},
                          {"type":"G4", "elements":"Siemens","eta":2.56450515, "gamma":8.04475192, "zeta": 2.5356155},
                          {"type":"G6", "elements":"Siemens","eta":2.33043463, "gamma": 0.50753377, "zeta": 0.93372721}
                         ]}
        finpr=exponential_new.FingerprintCalculator(cutoff=4001,Gs=Gs,Rct=3000,delta_R=100,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)
        
        # MAKE PREDICTIONS
        position=pos_array;
        neigh=nl.calculate(turb*num_turbs_predict,position)
        fingerprints=np.array(finpr.calculate(turb*num_turbs_predict,position,neigh,symbol))
        refwind,refstdev=self.model.predict(fingerprints)
        #print(neigh,fingerprints,refwind)
        
        return refwind

