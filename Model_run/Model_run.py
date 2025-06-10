# %%
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime as dt
import numpy.testing as npt
import numpy as np
import scipy as sp
from numpy import vstack, add, float64, multiply, exp, zeros, nan_to_num, vstack, where, power
from numpy import add, float64, multiply, exp, zeros, nan_to_num, vstack, where, power
import scipy.stats
from scipy.stats import gamma, beta
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numpy import sin, arange, isclose
from scipy.optimize import differential_evolution, Bounds, shgo, dual_annealing, minimize
from scipy import signal
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from importlib import reload
from matplotlib.patches import Patch
from mpl_toolkits import mplot3d
from matplotlib import cm, colors 
import warnings  
warnings.filterwarnings("ignore")

import os

#from statsmodels.graphics.tsaplots import plot_acf
#Set the directory
os.chdir("./Tracer_Transport_Model/Model_Run")
saveDir = './Tracer_Transport_Model/Output/Plots'

# Lets import the model and objective functions 
import tracer_models as tracer_models
from tracer_models import ObjectiveFunction
from tracer_models import Tracer_Mod_Wetness # Model which SAS parameter is function of Soil Wetness
from tracer_models import Tracer_Mod_Ptresh_Wetness # Model which SAS parameter is function of Soil Wetness and Precipitation Threshold

# %%

## Model calibration and validation
#Read Hyrological Data 
meas_df =  pd.read_csv("DATA/read_your_data", parse_dates=['Date'],index_col='Date',).squeeze("columns")
#interpolTE NAN values meas_df PET


wp =365    ## warm up period
nt = meas_df.Qmmd.size ## calibration period end index


## lets prepare the data to numpy as input format to run the model
evap =meas_df.PET[0:nt].to_numpy()
#apend zero to evap < 0
evap = np.where(evap < 0, 0, evap)
prec = meas_df.Pmmd[0:nt].to_numpy()
temp = meas_df.T_mean[0:nt].to_numpy()
flow = meas_df.Qmmd[0:nt].to_numpy()

##Oxygen isotpe data  from precipitation, P and streamflow, Q
ConP_O= meas_df.P_O[0:nt ].to_numpy()
ConQ_O= meas_df.Q_O[0:nt ].to_numpy()
Con_O_week = meas_df.Oxygen[0:2000].to_numpy() 

#nan values in ConP and ConQ to zero
ConP = np.nan_to_num(ConP_O)
ConQ = np.nan_to_num(ConQ_O)
Conweek = np.nan_to_num(Con_O_week)
#ConQ zero values to nan
ConQ = np.where(ConQ == 0, np.nan, ConQ)
Conweek = np.where(Conweek == 0, np.nan, Conweek)


#Lets  calculate 3 month moving average for the flow data to get Runoff Coefficient RC for calibration period

df = pd.DataFrame({"flow":meas_df.Qmmd[wp:nt] , "prec": meas_df.Pmmd[wp:nt]})

df_n  = df.groupby(np.arange(len(df))//90).mean()
df_n['RC'] = df_n.flow/df_n.prec

##  Prepare the inital storage Tracer composition for  unsaturated zone Su and Passive Groundwater Ssp
valid_indices_O = ~np.isnan(ConQ_O)
filtered_Qo_O = flow[valid_indices_O]
filtered_ConcQ_O = ConQ_O[valid_indices_O]
percentile_5_O = np.percentile(filtered_Qo_O, 5)
filtered_ConcQ_O = filtered_ConcQ_O[filtered_Qo_O <= percentile_5_O]
ConcSsp_avg_O = np.nanmean(filtered_ConcQ_O)
ConcSu_avg = np.nansum(ConP_O*prec) / np.nansum(prec)

# %%
##############################
# Calibrate the model
# Set the case as "Cal" ### 
# If the Case is "Cal" the model will retuns 
# Flux matrix of Fluxes as
# Flux_df = pd.DataFrame({'Flux_RM': Flux_RM, 'Flux_Pe': Flux_Pe, 'Flux_Ei': Flux_Ei, 'Flux_rf': Flux_rf, 'Flux_ea': Flux_ea, 'Flux_rs': Flux_rs, 'Flux_rfs': Flux_rfs, 'Flux_rfn': Flux_rfn, 'Flux_Qo': Flux_Qo, 'Flux_Qf': Flux_Qf, 
# 'Flux_Qof': Flux_Qof, 'Flux_Qstot': Flux_Qstot, 'Flux_Ql': Flux_Ql})
# Age matix of Fluxes as 
# Age_MR = [RM_Age_MR, Pe_Age_MR, rf_Age_MR, rfs_Age_MR, rfn_Age_MR, Qf_Age_MR, Qof_Age_MR, Qstot_Age_MR, Ql_Age_MR, Qo_Age_MR, rs_Age_MR, Ei_Age_MR, ea_Age_MR]
##############################
# ## df_WB ## will return the hydrological model water balance components   
# df_WB = ["Qtot", "Qs", "Ql", "Qf", "Qof", "Qo", "Ss", "Si", "Sr", "Ssa", "Ssp", "Sf", "Rs","Rsg", "Rmelt", "Rff", "Rfn", "Rfs", "Ei", "Ea", "Etot", "Pe",
#                   "Ps", "M", "Cr","Cn", "alphat"]
############################## ----Define the Model Parameter----#################################
parameters = pd.DataFrame(columns=["initial", "pmin", "pmax"])
parameters.loc["srmax"] = (300.0, 100, 500)           #0 SU capacity 
parameters.loc["lp"] =    (0.26, 1e-5, 1)             #1 Ea calculation parameter
parameters.loc["Rsmax"] =  (1.05, 0.0, 1.2)          #2 Saturated hydraulic conductivity
parameters.loc["beta"] = (0.17, 1e-5, 5.0)           #3 Parameter determining the nonlinearity of outflow / recharge.
parameters.loc["Kp"] =  (1e-8, 1e-10, 1e-5)         #4 Loss coefficient
parameters.loc["Imax"] = (4.97, 1.2, 5.0)            #5 Interception capacity
parameters.loc["Ka"] = (0.18, 0.01, 0.2)             #6 slow responce coefficient
parameters.loc["tt"] = (-2.0, -4.0, 5.0)             #7 Melt treshold temperature
parameters.loc["fmelt"] =  (2.81, 1.0, 5.0)           #8 melt factor
parameters.loc["sfmax"] = (1.013, 1.0, 20.0)          #9 Fast responce capacity
parameters.loc["kf"] = (0.9, 0.01, 2.0)               #10 fast responce coefficient
parameters.loc["cp"] = (0.5, 0.1, 1.0)              #11 divider to fast responce slow responce (1-cp) =Rfs
parameters.loc["cn"] = (0.6, 0.1, 1.0)              #12 divider to fast responce to overland flow
parameters.loc["Bf"] = (1e-8, 1e-10,1e-5)            #13 overland flow coefficient
parameters.loc["Ssp"] = (3000.0, 500.0, 10000.0)      #14  passive storage  coefficient
parameters.loc["Ptres"] = (6.0, 2.0, 20.0)            #15  SAS  tree water
parameters.loc["SU_Alpha"] = (0.1, 0.01, 1.0)         #16  SAS  SU_alpha
parameters.loc["SG_Alpha"] = (0.99, 0.98, 1.0)        #17 SAS SG_alpha  

par =parameters.initial.values

# %%
## get fittenss
##  Change the Tracer_Mod_Wetness to Tracer_Mod_Ptresh_Wetness, to use the model with Precipitation Threshold
def fitness(par):
    par =par
    Flux_TB, Agn_MR , df_WB = Tracer_Mod_Wetness.Tracer_Mod().get_tracer_balance(prec, evap, temp, ConP_O, ConcSsp_avg_O, ConcSu_avg,  p = par,  Case ='Cal') #
    Flux_Qtot  = Flux_TB.Flux_Qstot+ Flux_TB.Flux_Qo + Flux_TB.Flux_Qof + Flux_TB.Flux_Qf # get total Q flux
    Qtot_Age  = Agn_MR[5] + Agn_MR[6] + Agn_MR[7]+ Agn_MR[9] # get Qtot _Age  = Qf_Age_MR, Qof_Age_MR, Qstot_Age_MR,
    Qtot = Qtot_Age.sum(axis=0) #Sum Qtot_Age columns to estimate total Q mmd-1
    Flux_Qtot = Flux_Qtot[:-1] # remove the last value as it is extra
    Conc_sim  = Flux_Qtot/Qtot[:-1] # calculate the concentration of the simulated tracer
    C_obs = ConQ_O[wp:nt] # observed tracer concentration
    C_sim  = Conc_sim[wp:nt].values # simulated tracer concentration for defined calibration period 
    C_week = Conweek[wp:nt] #Observed weekly tracer concentration 
    obs = flow[wp:nt] # observed flow for defined calibration period
    RC = df_n.RC # observed runoff coefficient
    sim_Q = df_WB.Qtot[wp:nt].values # simulated flow for defined calibration period
    obs_Q = flow[wp:nt] # observed flow for defined calibration period
    Q_Avarage_sim  = pd.DataFrame({"flow":df_WB.Qtot[wp:nt] , "prec":prec[wp:nt]}) # Claculate the average flow and precipitation for the defined calibration period
    df_RC = Q_Avarage_sim.groupby(np.arange(len(Q_Avarage_sim))//90).mean() # 
    RC_sim = df_RC.flow/df_RC.prec # simulated runoff coefficient for the defined calibration period
    nse_Q =  ObjectiveFunction.nashsutcliffe(obs_Q, sim_Q) # Nash-Sutcliffe efficiency for flow
    nse_QO = ObjectiveFunction.nashsutcliffe(C_obs,C_sim) # Nash-Sutcliffe efficiency for tracer balance
    nse_QO_week = ObjectiveFunction.mse(C_week, C_sim) # Mean Squared Error for weekly tracer concentration
    nse_logQ = ObjectiveFunction.lognashsutcliffe(obs,sim_Q) # Log Nash-Sutcliffe efficiency for flow
    nse_FDC =  ObjectiveFunction.nse_FDC(obs,sim_Q) # Nash-Sutcliffe efficiency for flow duration curve
    nse_RC = ObjectiveFunction.nashsutcliffe(RC,RC_sim) # Nash-Sutcliffe efficiency for runoff coefficient
    # Calculate the fitness value DE
    DE = np.sqrt(0.5*(((0.25*(1 -nse_Q)**2 + 0.25*(1-nse_logQ)**2 + 0.25*(1-nse_FDC)**2 +0.25*(1-nse_RC)**2)) + (0.5*(1-nse_QO)**2 + 0.5*(nse_QO_week)**2))) # Cost function
    return DE

bounds = Bounds(lb= parameters.pmin.values, ub=parameters.pmax.values, keep_feasible=False)

# Model to be optimized
# Initialize a list to hold solutions that meet the criteria
satisfactory_solutions = []

# Maximum number of solutions to keep
max_solutions = 1000  # Adjust as needed

def callback_efficient(xk, convergence):
    global satisfactory_solutions
    # Check if the solution meets the criterion
    current_fitness = fitness(xk)
    if current_fitness < 1:  # Criterion: fitness value should be less than 1
        satisfactory_solutions.append((xk, current_fitness))
        # Sort the list based on fitness (for minimizing)
        satisfactory_solutions.sort(key=lambda sol: sol[1])
        # Ensure the list does not exceed the maximum length
        satisfactory_solutions = satisfactory_solutions[:max_solutions]

# %%
# Run the differential evolution optimizer set the workers as nedeed
# Change the updating parameter to 'differ" to change the updating strategy
#fitness is your fittnes function valu of "DE" 
result = differential_evolution(fitness, bounds,  callback=callback_efficient, updating='immediate', workers=8, maxiter=500)

print("optimization done!!!!")

# Save Result .x as data frame 
df_result = pd.DataFrame({"par":result.x}) 
df_result.to_csv(".Set_Directory_to_save")

# %%
###### Model Comparison with optimized parameters ######
#Read_optimized parameter function of soil wetness to run the model 
Parameter_S1 = pd.read_csv(".read_the_optimized_parameter_model_calibration_S1.csv")
parOpt_S1 = Parameter_S1.par.values

#Read_optimized parameter  Ptresh and wetness to run the model 
Parameter_S2 = pd.read_csv(".read_the_optimized_parameter_model_calibration_S2.csv")
parOpt_S2 = Parameter_S2.par.values

################Run the model with calibrated parameters ################
# Set the case as "notCal" to retun the Storage age matrix and Flux matrix
# Storage_Age_MR =  [SS_Age_MR, Si_Age_MR, Sr_Age_MR, Sf_Age_MR, SSa_Age_MR]
# Storage_Flux_MR = [SS_Flux_MR, Si_Flux_MR, Sr_Flux_MR, Sf_Flux_MR, SSa_Flux_MR]

Agn_MR_S1, Flux_dfn_S1, Storage_Age_MR_S1, Storage_Flux_MR_S1, TrB_S1, WB_S1, = Tracer_Mod_Wetness.Tracer_Mod().get_tracer_balance(prec, evap, temp, ConP_O,  ConcSsp_avg_O, ConcSu_avg,  p = parOpt_S1, Case ='notCal')
Agn_MR_S2, Flux_dfn_S2, Storage_Age_MR_S2, Storage_Flux_MR_S2, TrB_S2, WB_S2 =Tracer_Mod_Ptresh_Wetness.Tracer_Mod().get_tracer_balance(prec, evap, temp, ConP_O, ConcSsp_avg_O, ConcSu_avg,  p = parOpt_S2,  Case ='notCal')

# Check the result for water balance
def Get_WB_error(WB_df):
    #Snow water balance 
    Ss_dif = WB_df.Ss.diff()
    # Remove the firts raw SSdif and index zero 
    WB_ss  = WB_df.Ps[:-1] - WB_df.M[:-1] -  Ss_dif[1:].values

    #Interseption water balance 
    Si_dif = WB_df.Si.diff()
    # Remove the firts raw SSdif and index zero
    WB_si  = WB_df.Rmelt[:-1] - WB_df.Pe[:-1] - WB_df.Ei[:-1]- Si_dif[1:].values

    #Root zone water balance
    Sr_dif = WB_df.Sr.diff()
    # Remove the firts raw SSdif and index zero
    WB_sr  = WB_df.Pe[:-1] - WB_df.Ea[:-1] - WB_df.Rff[:-1] - WB_df.Rfs[:-1]- WB_df.Rs[:-1] - Sr_dif[1:].values

    #Fast responce Water balance
    Sf_dif = WB_df.Sf.diff()
    # Remove the firts raw SSdif and index zero
    WB_sf  = WB_df.Rfn[:-1] - WB_df.Qof[:-1] - WB_df.Qf[:-1] - Sf_dif[1:].values

    # Slow responce Water balance
    GW_dif = (WB_df.Ssa+WB_df.Ssp).diff()
    # Remove the firts raw SSdif and index zero
    WB_gw  = WB_df.Rfs[:-1] + WB_df.Rs[:-1]  - WB_df.Ql[:-1] - WB_df.Qs[:-1] - GW_dif[1:].values

    #WB eror data frame WB_ss, WB_si, WB_sr, WB_sf, WB_gw
    WB_error = pd.DataFrame({"WB_ss":WB_ss, "WB_si":WB_si, "WB_sr":WB_sr, "WB_sf":WB_sf, "WB_gw":WB_gw})
    
    return WB_error

# get the result # Run the optimised model for validation period

Error_Wb = Get_WB_error(WB_S2)
#tidy the 
Error_long = pd.melt(Error_Wb, value_vars=['WB_sr', 'WB_si', 'WB_ss', 'WB_sf', 'WB_gw'],value_name='Error', ignore_index=False)
#plot Error long facet grid
g = sns.FacetGrid(Error_long, col="variable", col_wrap=3, sharex=False, sharey=False)
g = g.map(plt.hist, "Error", bins=50)
# get the result # Run the optimised model for validation period
WB_S2.index  = meas_df.index[0:nt]
WB_S2['date'] = WB_S2.index.map(lambda x: x.strftime('%Y-%m-%d'))
Evaluation_Q_S2  = pd.DataFrame(ObjectiveFunction.calculate_all_functions(flow[wp:nt], WB_S2.Qtot[wp:nt]))

# %%
## Get objective function for streamflow
WB_S1.index  = meas_df.index[0:nt]
Evaluation_S1 = pd.DataFrame(ObjectiveFunction.calculate_all_functions(flow[wp:nt], WB_S1.Qtot[wp:nt]))
## Get objective function for FDC 
sort_obs = np.sort(meas_df.Qmmd[~np.isnan(meas_df.Qmmd)])[::-1]
exceedence_obs = np.arange(1.,len(sort_obs)+1) / len(sort_obs)
sort_sim_S2 = np.sort(WB_S2.Qtot[wp:nt])[::-1]
sort_sim_S1 = np.sort(WB_S1.Qtot[wp:nt])[::-1]
exceedence_sim_S2 = np.arange(1.,len(sort_sim_S2)+1) / len(sort_sim_S2)
exceedence_sim_S1 = np.arange(1.,len(sort_sim_S1)+1) / len(sort_sim_S1)
# Get objective function for runoff coefficient
##prepare data to get meman values 
dfRC = pd.DataFrame({"flow":meas_df.Qmmd[wp:nt] , "prec":prec[wp:nt]})
#set index date
dfRC.index = meas_df.index[wp:nt]
#group by 90 days keep date get mean values
dfRC = dfRC.groupby(pd.Grouper(freq='90D')).mean()
dfRC['RC'] = dfRC.flow/dfRC.prec
df_sim_S2 = pd.DataFrame({"flow":WB_S2.Qtot[wp:nt] , "prec":prec[wp:nt]})
df_sim_S2.index = WB_S2.index[wp:nt]
df_sim_S2 = df_sim_S2.groupby(pd.Grouper(freq='90D')).mean()
df_sim_S2['RC'] = df_sim_S2.flow/df_sim_S2.prec
df_sim_S1  = pd.DataFrame({"flow":WB_S1.Qtot[wp:nt] , "prec":prec[wp:nt]})
df_sim_S1.index = WB_S1.index[wp:nt]
df_sim_S1 = df_sim_S1.groupby(pd.Grouper(freq='90D')).mean()
df_sim_S1['RC'] = df_sim_S1.flow/df_sim_S1.prec

ObjectiveFunction.calculate_all_functions(dfRC.RC, df_sim_S1.RC)
ObjectiveFunction.calculate_all_functions(dfRC.RC, df_sim_S2.RC)

## now lets get the tracer balance for S1 and S2 
#get Qtotal age matrix for S1 and S2
Qtot_Age_S2  = Agn_MR_S2[5] + Agn_MR_S2[6]  + Agn_MR_S2[7] + Agn_MR_S2[9]
Qtot_S2 = Qtot_Age_S2.sum(axis=0)
WB_Qtot_S2 = WB_S2.Qtot[wp:nt].values
# get fluc matric which is  QxC
Flux_Qtot_S2  = Flux_dfn_S2.Flux_Qstot+ Flux_dfn_S2.Flux_Qo + Flux_dfn_S2.Flux_Qof + Flux_dfn_S2.Flux_Qf
#Skip last row Flux_Qtot
Conc_sim_S2  = Flux_Qtot_S2[:-1]/Qtot_S2[:-1]
C_obs = ConQ[wp:nt]
C_sim_S2 = Conc_sim_S2[wp:nt].values
TB_nse_S2 = ObjectiveFunction.calculate_all_functions(C_obs,C_sim_S2)
# ## Lets get same calcultions above for S1
Qtot_Age_S1  = Agn_MR_S1[5] + Agn_MR_S1[6]  + Agn_MR_S1[7] + Agn_MR_S1[9]
#Sum Qtot_Age columns
Qtot_S1 = Qtot_Age_S1.sum(axis=0)
WB_Qtot_S1 = WB_S1.Qtot[wp:nt].values
Flux_Qtot_S1  = Flux_dfn_S1.Flux_Qstot+ Flux_dfn_S1.Flux_Qo + Flux_dfn_S1.Flux_Qof + Flux_dfn_S1.Flux_Qf
#Skip last row Flux_Qtot
Conc_sim_S1  = Flux_Qtot_S1[:-1]/Qtot_S1[:-1]
C_obs= ConQ[wp:nt]
C_sim_S1  = Conc_sim_S1[wp:nt].values
TB_nse_S1 = ObjectiveFunction.calculate_all_functions(C_obs,C_sim_S1)

# GET RF from the model 
Q_rf_S2 = Agn_MR_S2[2].sum(axis=0)
Q_rf_S1 = Agn_MR_S1[2].sum(axis=0)
#Qf = Sim_df.Qf.values
Con_Qf_S2 = Flux_dfn_S2.Flux_Qf[:-1]/Q_rf_S2[:-1]
Con_Qf_S1 = Flux_dfn_S1.Flux_Qf[:-1]/Q_rf_S1[:-1]

## Lets get the transit time estimation TTD for S2 and S1
# %%
######### Define the calibtation period for TTD calculation #########
Tr_length = nt # length of the calibration period
TTD_Qtot_S2=np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) 
for i in range(0,Tr_length):
    TTD_Qtot_S2[:,i] = Qtot_Age_S2[:,i]/np.sum(Qtot_Age_S2[:,i])
TTD_Qtot_S2[:,Tr_length] = Qtot_Age_S2[:,Tr_length-1]/np.sum(Qtot_Age_S2[:,Tr_length-1])
#Lets Calulate the TTD for SASt
TTD_Qtot_S1=np.zeros((Tr_length+1,Tr_length+1),  dtype=float64)
for i in range(0,Tr_length):
    TTD_Qtot_S1[:,i] = Qtot_Age_S1[:,i]/np.sum(Qtot_Age_S1[:,i])
TTD_Qtot_S1[:,Tr_length] = Qtot_Age_S1[:,Tr_length-1]/np.sum(Qtot_Age_S1[:,Tr_length-1])
#Lets get TTD_for Rf for S2 and S1
TTD_rf_S1=np.zeros((Tr_length+1,Tr_length+1),  dtype=float64)
TTD_rf_S2=np.zeros((Tr_length+1,Tr_length+1),  dtype=float64)
Qrf_tot_Age_S1 = Agn_MR_S1[2]# Agn_MR_SASt[5] + Agn_MR_SASt[6]  + Agn_MR_SASt[9]
Qrf_tot_Age_S2 = Agn_MR_S2[2] # Agn_MR[5] + Agn_MR[6]  + Agn_MR[9]
for i in range(0,Tr_length):
    TTD_rf_S1[:,i] = Qrf_tot_Age_S1[:,i]/np.sum(Qrf_tot_Age_S1[:,i])
    TTD_rf_S2[:,i] = Qrf_tot_Age_S2[:,i]/np.sum(Qrf_tot_Age_S2[:,i])
TTD_rf_S1[:,Tr_length] =Qrf_tot_Age_S2[:,Tr_length-1]/np.sum(Qrf_tot_Age_S2[:,Tr_length-1])
TTD_rf_S2[:,Tr_length] = Qrf_tot_Age_S1[:,Tr_length-1]/np.sum(Qrf_tot_Age_S1[:,Tr_length-1])

# %%
#lets prpare the data for correlation plots with precipitation and streamflow and Soil moisture
# normilize measure_df prec between 0 and 1
norm_prec = (meas_df['Prec'][wp:nt] - meas_df['Prec'][wp:nt].min()) / (meas_df['Prec'][wp:nt].max() - meas_df['Prec'][wp:nt].min())
# normilize measure_df prec between 0 and 1
norm_Q = (meas_df['Qmmd'][wp:nt] - meas_df['Qmmd'][wp:nt].min()) / (meas_df['Qmmd'][wp:nt].max() - meas_df['Qmmd'][wp:nt].min())
norm_SM  = (meas_df['SWC_Avrg'][wp:nt] - meas_df['SWC_Avrg'][wp:nt].min()) / (meas_df['SWC_Avrg'][wp:nt].max() - meas_df['SWC_Avrg'][wp:nt].min())
#Set index for all to the date
norm_prec.index = meas_df.index[wp:nt]
norm_Q.index = meas_df.index[wp:nt]
norm_SM.index = meas_df.index[wp:nt]

######## lets calculate the fraction young water  for T<2, 7, 90, 365 and old as 
fyw_S2_0_2 = np.sum(TTD_Qtot_S2[0:2,], axis=0)*100
fyw_S2_0_7 =  np.sum(TTD_Qtot_S2[0:7,], axis=0)*100
fyw_S2_0_90 =  np.sum(TTD_Qtot_S2[0:90,], axis=0)*100
fyw_S2_7_90 =  np.sum(TTD_Qtot_S2[7:90,], axis=0)*100
fyw_S2_90_365 =  np.sum(TTD_Qtot_S2[90:365,], axis=0)*100
Fyw_S1_0_7 =  np.sum(TTD_Qtot_S1[0:7,], axis=0)*100
Fyw_S1_0_2 =  np.sum(TTD_Qtot_S1[0:2,], axis=0)*100
Fyw_S1_7_90 =  np.sum(TTD_Qtot_S1[7:90,], axis=0)*100
Fyw_S1_90_365 =  np.sum(TTD_Qtot_S1[90:365,], axis=0)*100
Fyw_S1_0_90= np.sum(TTD_Qtot_S1[0:90,], axis=0)*100
#Eliminate fye_365 >0.1 to NAN

#Append it df_fyw
df_fyw_S2= pd.DataFrame({'Date': meas_df.index[wp:nt], 'Qmmd':  meas_df.Qmmd[wp:nt], 'Qnorm': norm_Q, 'Pmmd': meas_df.Prec[wp:nt], 'Pnorm':norm_prec, 'SM':norm_SM, 'Season': meas_df.Season[wp:nt], 
                       'Season_M':meas_df.Season_M[wp:nt], 'SM_Meas':meas_df.SWC_Avrg[wp:nt],'0_2': fyw_S2_0_2[wp:nt], '7_90': fyw_S2_7_90[wp:nt], '0_7': fyw_S2_0_7[wp:nt], 
                       '90_365': fyw_S2_90_365[wp:nt],  'Fyw_90 [-]': fyw_S2_0_90[wp:nt]})
#Set Date as index
#df_fyw = df_fyw.set_index('Date')
df_fyw_S2['case'] = 'S2'
# 
df_fyw_SASt = pd.DataFrame({'Date': meas_df.index[wp:nt],  'Qmmd':  meas_df.Qmmd[wp:nt], 'Qnorm': norm_Q, 'Pmmd': meas_df.Prec[wp:nt], 'Pnorm':norm_prec, 'SM':norm_SM, 'Season': meas_df.Season[wp:nt], 
                            'Season_M':meas_df.Season_M[wp:nt], 'SM_Meas':meas_df.SWC_Avrg[wp:nt], '0_2': Fyw_S1_0_2[wp:nt],'7_90': Fyw_S1_7_90[wp:nt], '0_7': Fyw_S1_0_7[wp:nt],
                            '90_365':Fyw_S1_90_365[wp:nt], 'Fyw_90 [-]':Fyw_S1_0_90[wp:nt]})

