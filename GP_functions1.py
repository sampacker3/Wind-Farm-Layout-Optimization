import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
from IPython.display import display
import three_desc_model as exponential_new
from cutoffs import Polynomial
GPy.plotting.change_plotting_library('matplotlib')

class TurbineUtility:
#    @staticmethod
    def __init__(self):
        self.symbol = "Siemens"
        # INITIALISE NEIGHBOUR LIST CALCULATOR
        self.nl=exponential_new.NeighborlistCalculator(cutoff=4001,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)
            # INITIALISE FINGERPRINT CALCULATOR
        self.Gs = {self.symbol: TurbineUtility.load_turbine_parameters(self.symbol)}
        self.finpr=exponential_new.FingerprintCalculator(cutoff=4001,Gs=self.Gs,Rct=3000,delta_R=100,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)
        return
    
    def load_turbine_parameters(turbine_type):
        if turbine_type == "Siemens":
            return [{"type":"G2", "turbine":"Siemens","eta":4.25387599, "offset": 1.0151402},
                    {"type":"G4", "elements":"Siemens","eta":2.56450515, "gamma":8.04475192, "zeta": 2.5356155},
                    {"type":"G6", "elements":"Siemens","eta":2.33043463, "gamma": 0.50753377, "zeta": 0.93372721}
                   ]
        else:
            pass
        
    def nlist(self,pos_array,num_turbs_predict):
        # DECLARE TURBINE TYPE
        turb = [self.symbol]

        # Calculate Neighbourlist
        neigh=self.nl.calculate(turb*num_turbs_predict,pos_array)
        return neigh 
    
    def fingerprint(self,pos_array,num_turbs_predict):
            # DECLARE TURBINE TYPE
        turb = [self.symbol]
            # Calculate neighbourlist
        neigh = self.nlist(pos_array,num_turbs_predict)
            # Calculate fingerprint
        fingerprints=np.array(self.finpr.calculate(turb*num_turbs_predict,pos_array,neigh,self.symbol))
        return fingerprints
        




class GP_train:
    def __init__(self):
        self.turbine_utility = TurbineUtility()
        return

    def train_model(self):
        # LOAD DATA
        dataset_full = pd.read_csv("turb_data.csv", index_col=0)
        symbol="Siemens"
        k=0
        count=0
        
        numsims=dataset_full["IDnum"].iloc[-1]+1
        turb = [symbol]
        # USE TurbineUtility here
        # INITIALISE NEIGHBOUR LIST
        nl = self.turbine_utility.nl
        Gs = self.turbine_utility.Gs
        finpr = self.turbine_utility.finpr
        pass
            
         # CALCULATE NL AND FINGPR  
         # outputs: fingerprints and reference velocities
        count=0
        dataset_fp=np.empty(shape=(0, 3))
        dataset_rws=np.empty(shape=(0, 1))
        dataset_ws=np.empty(shape=(0, 1))
        dataset_pos=np.empty(shape=(0, 2))
        for i in range(numsims):
            numturb=dataset_full["num_tot_turb"].iloc[count]
            position = np.empty((numturb,2))
            rws = np.empty((numturb,1))
            ws = np.empty((numturb,1))
            fp = np.empty((numturb,3))
            for k in range(numturb):
                position[k,0]=dataset["x_coord"].iloc[count]#dataset_full.at[count,"x_coord"]
                position[k,1]=dataset["y_coord"].iloc[count]#dataset_full.at[count,"y_coord"]
                rws[k,0]=dataset["ref_wind_speed"].iloc[count]#dataset_full.at[count,"ref_wind_speed"]
                ws[k,0]=dataset["wind_speed"].iloc[count]#dataset_full.at[count,"wind_speed"]
                count = count+ 1

            neigh=self.turbine_utility.nlist(position, numturb) #And here
            fingerprints=self.turbine_utility.fingerprint(position, numturb) # and here
            fingerprints=np.array(fingerprints)
            dataset_fp=np.append(dataset_fp,fingerprints,axis=0)
            dataset_ws=np.append(dataset_ws,ws,axis=0)
            dataset_rws=np.append(dataset_rws,rws,axis=0)
            dataset_pos=np.append(dataset_pos,position,axis=0)
            dataset=np.concatenate((dataset_fp, dataset_ws),axis=1)


        all_data=np.concatenate((dataset_fp, dataset_ws, dataset_rws),axis=1)
        all_data = pd.DataFrame(all_data, columns = ['Fingerprint(G2)','Fingerprint(G4)','Fingerprint(G6)','Wind_Speed','Ref_Wind_Speed'])
        X=all_data[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)","Wind_Speed"]].to_numpy()
        Y=all_data[["Ref_Wind_Speed"]].to_numpy()
        train_dataset = all_data.sample(frac=0.8, random_state=0)
        test_dataset = all_data.drop(train_dataset.index)
        Xtrain=train_dataset[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)","Wind_Speed"]].to_numpy()
        Ytrain=train_dataset[["Ref_Wind_Speed"]].to_numpy()
        Xtest=test_dataset[["Fingerprint(G2)","Fingerprint(G4)","Fingerprint(G6)","Wind_Speed"]].to_numpy()
        Ytest=test_dataset[["Ref_Wind_Speed"]].to_numpy()

        
        # DEFINE KERNEL
        ker = GPy.kern.RBF(4,lengthscale=0.1, ARD=True)
    
        # CREATE GP MODEL
        m = GPy.models.GPRegression(Xtrain,Ytrain,ker)
       
        # OPTIMISE
        m.optimize(messages=True,max_f_eval = 1000)
        return m 

    
        # CLASS FOR PREDICTING,
class GP_predict:
    def __init__(self, model):
        self.model = model
        self.tu = TurbineUtility()
        #self.symbol = "Siemens"
        # INITIALISE NEIGHBOUR LIST CALCULATOR
        #self.nl=exponential_new.NeighborlistCalculator(cutoff=4001,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)
            # INITIALISE FINGERPRINT CALCULATOR
        #self.Gs = {self.symbol: TurbineUtility.load_turbine_parameters(self.symbol)}
        #self.finpr=exponential_new.FingerprintCalculator(cutoff=4001,Gs=self.Gs,Rct=3000,delta_R=100,cone_grad=0.12582561117875557, cone_offset=72.24947126849844)
        return
    
    def predict(self, pos_array, num_turbs_predict):
            # DECLARE TURBINE TYPE
        turb = [self.tu.symbol]
        
        # MAKE PREDICTIONS
        position = pos_array
            # Calculate fingerprint
        fingerprints = self.tu.fingerprint(position, num_turbs_predict)
            # Predict wind speed
        refwind, refstdev = self.model.predict(fingerprints)
            
        return refwind 
