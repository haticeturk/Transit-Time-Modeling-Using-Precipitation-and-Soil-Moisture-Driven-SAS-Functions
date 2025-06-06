# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt
import numpy.testing as npt
import numpy as np
import logging
import scipy
import warnings
from logging import getLogger
from pandas import DataFrame
from numpy import vstack, add, float64, multiply, exp, zeros, nan_to_num, vstack, where, power
from numpy import add, float64, multiply, exp, zeros, nan_to_num, vstack, where, power
import scipy.stats
from scipy.stats import gamma, beta
from numpy import add, float64, multiply, exp, zeros, nan_to_num, vstack, where, power
from pandas import DataFrame
from numpy import sin, arange, isclose
from scipy import signal
warnings.filterwarnings("ignore")

#plot the results
from .decorators import njit
from .utils import check_numba
logger = getLogger(__name__)

# %%
class Tracer_Mod(object):
    _name = "Tracer_Mod"

    def Tracer_Tracking(self, prec, evap, temp, ConP,  ConAvg,  ConSuAvg, p,  **kwargs):
        #Tracking  length  as lenght pr as float 
        np.seterr(all="ignore")
        Tr_length = prec.size
        x = [] 
        x=np.array(x)
        dt  = 1.0
       #variable DEFINITION
        # All Age_MR represent Age matrix (Q(i)xTTD(i)) for each flux for traxing period (Tr_lengthxTr_length)
        # and storage wgich is used to calculate the age distribution of the water in each compartment for all traking period
        # All Flux_MR represent Flux matrix (Q(i)xConc(i)TTD(i)) for each flux it will be (Tr_lengthxTr_length) 
        # and storage which is used to calculate the concentration of the water in each compartment
        # All Conc_MR represent Concentration matrix (Conc(i)TTD(i)) for each flux and storage which is used to calculate the concentration of the water in each compartment
        # All Flux represent total flux (Summ of Flux_MR matrix)x used to calculate the total flux of the water in each compartment
        # All Storage_Age_MR represent Age matrix of the storage (RTD) for each storage for traxing period (Tr_lengthxTr_length)
        ##
        #Lets start with the snow bucket
        ## Hydrological Model get the water balance components
        #Snow bucket, ss is snow storage, ps precipitaation as snow, m is meltwater
        ep = evap # PET

        ss, ps, m = self.get_snow_balance(prec=prec, temp=temp,tt=p[7], fmelt=p[8])
        
        pr = prec - ps # Remove snowfall from precipitation

        rmelt = pr + m  # ad melt to precipitation to get total rain and melt
        

        #Tracer balance in snow bucket example 
        #RM_Age_MR is is the Age matrix for the rain and melt water in the snow bucket (Tr_lengthxTr_length)
        #Conc_Age_MR is the Concentration matrix for the rain and melt water in the snow bucket(Tr_lengthxTr_length)
        #RM_flux_MR is the Flux matrix for the rain and melt water in the snow bucket(Conc_Age_MR * RM_Age_MR)
        #Flux_RM is the total flux of the water out of the snow bucket (sum of RM_flux_MR  for each column0
        #SS_Age_MR is the Age matrix for the stored water in the snow bucket (Tr_lengthxTr_length)
        #SS_Flux_MR is the Flux matrix for the stored water in the snow bucket (Conc_Age_MR * SS_Age_MR)
                                                                
        RM_Age_MR, RM_flux_MR, Conc_Age_MR, Flux_RM, SS_Age_MR, SS_Flux_MR = self.Tracer_sampling_snow(prec,  ConP, rmelt, Tr_length, 1.0, 1.0)


        #Interception bucket 
        si, ei, pe = self.get_interception_balance(pr=rmelt, ep=ep, Imax=p[5])

        #Tracer balance in interception bucket
        Pe_Age_MR, Pe_Flux_MR, Pe_Conc_MR, Ei_Age_MR, Ei_Flux_MR, Si_Age_MR, Si_Flux_MR,  Flux_Pe, Flux_Ei, TB_si = self.Tracer_sampling_Si(RM_Age_MR, Conc_Age_MR, pe, ei,  Tr_length, 1.0, 1.0, x)


        #Soil bucket
        ep = ep - ei  # Update potential evaporation after interception
        sr, rs, ea, rff, rfs, pe, cr,  alphat  = self.get_root_zone_balance(pr, pe, ep, srmax=p[0], lp=p[1], Rsmax=p[2], beta=p[3], dt=dt, cp = p[11], alpha_Su = p[16], Ptresh=p[15])

        #Tracer balance in soil bucket
        #Time variable SAS_alpha parameter 

        rtot =  rff + rfs 
        rf_Age_MR, rf_Conc_MR, Flux_rf, ea_Age_MR, ea_Conc_MR, Flux_ea, rs_Age_MR, rs_Conc_MR, Flux_rs, Sr_Age_MR, Sr_Flux_MR, TB_Sr = self.Tracer_sampling_SR(Pe_Age_MR, Pe_Conc_MR, rtot, ea, rs, Tr_length, alphat, 1.0, x, p[0], ConSuAvg)

        Qo, rfn, Cn = self.get_overland_flow(rmelt, rff, Ptresh = p[15],  cn = p[12])
        ep = ep - ea  # Update potential evaporation after interception

 
        #Tracer balance in  preferential flow divitions to fast responce, recharge to groundwater and Overland flow
        rfs_Age_MR, rfs_Conc_MR, rfs_Flux_MR, Flux_rfs, rfn_Age_MR, rfn_Conc_MR, rfn_Flux_MR, Flux_rfn, Qo_Age_MR, Qo_Conc_MR, Qo_Flux_MR, Flux_Qo = self.Rf_division(rf_Age_MR, rf_Conc_MR, Tr_length, p[11] , p[12] )

        ##Fast responce bucket
        sf, Qf, Qof = self.get_fast_res_balance(rfn, sfmax=p[9], dt=dt,  Bf = p[13], kf=p[10])
        Qf_Age_MR, Qof_Age_MR, Sf_Age_MR, Sf_Conc_MR, Sf_Flux_MR, Flux_Qf, Flux_Qof, TB_sf = self.Tracer_sampling_Sf(rfn_Age_MR, rfn_Conc_MR, Qof, Qf, Tr_length, 1, 1, x, p[9])

          
        #Tracer balance GW
        rsg = rs+rfs  ## update recharge before storage
        ssa, Qs, Ql,ssp  = self.get_slow_res_balance(rsg, Ka=p[6], Kp= p[4], SSp = p[14], dt=1.0)

        #Sum all input to Groundwater 
        ConAvg = ConAvg
        Qin_Age_MR = rs_Age_MR +rfs_Age_MR
        Flux_Tot_MR = rs_Age_MR*rs_Conc_MR +rfs_Age_MR*rfs_Conc_MR
        Conc_in_MR = Flux_Tot_MR/Qin_Age_MR
        Conc_in_MR[Qin_Age_MR == 0.0] = 0.0

        Qstot = Qs                                                              

        Qstot_Age_MR, Flux_Qstot, SSa_Age_MR, SSa_Conc_MR, SSa_Flux_MR, TB_Ss, Qstot_Conc_MR, Ql_Age_MR, Ql_Conc_MR, Ql_Flux_MR, Flux_Ql= self.Tracer_sampling_SG(Qin_Age_MR, Conc_in_MR, Qstot, Ql, Tr_length, p[17], 1, x, p[14], ConAvg)

       # Create a array stores Age_MR and Conc_MR for each compartment
        Flux_df = pd.DataFrame({'Flux_RM': Flux_RM, 'Flux_Pe': Flux_Pe, 'Flux_Ei': Flux_Ei, 'Flux_rf': Flux_rf, 'Flux_ea': Flux_ea, 'Flux_rs': Flux_rs, 'Flux_rfs': Flux_rfs, 'Flux_rfn': Flux_rfn, 'Flux_Qo': Flux_Qo, 'Flux_Qf': Flux_Qf, 
                            'Flux_Qof': Flux_Qof, 'Flux_Qstot': Flux_Qstot, 'Flux_Ql': Flux_Ql})
        Age_MR = [RM_Age_MR, Pe_Age_MR, rf_Age_MR, rfs_Age_MR, rfn_Age_MR, Qf_Age_MR, Qof_Age_MR, Qstot_Age_MR, Ql_Age_MR, Qo_Age_MR, rs_Age_MR, Ei_Age_MR, ea_Age_MR]
        Storage_Age_MR =  [SS_Age_MR, Si_Age_MR, Sr_Age_MR, Sf_Age_MR, SSa_Age_MR]
        Storage_Flux_MR = [SS_Flux_MR, Si_Flux_MR, Sr_Flux_MR, Sf_Flux_MR, SSa_Flux_MR]
        TrB = pd.DataFrame({'TB_Si': TB_si, 'TB_Sr': TB_Sr, 'TB_sf': TB_sf, 'TB_Ss': TB_Ss})

        Qtot = Qs + Qf + Qof + Qo
        Etot = ea + ei 
        data = (Qtot, Qs, Ql, Qf, Qof, Qo, ss, si, sr, ssa, ssp, sf, rs, rsg, rmelt, rff, rfn, rfs, ei, ea, Etot, pe, ps, m, cr, Cn, alphat)
        columns = ["Qtot", "Qs", "Ql", "Qf", "Qof", "Qo", "Ss", "Si", "Sr", "Ssa", "Ssp", "Sf", "Rs","Rsg", "Rmelt", "Rff", "Rfn", "Rfs", "Ei", "Ea", "Etot", "Pe",
                  "Ps", "M", "Cr","Cn", "alphat"]

        WB_df = DataFrame(data=vstack(data).T, columns=columns)
        ## Tracer Balance Fast responce 

        return Age_MR, Flux_df, Storage_Age_MR, Storage_Flux_MR, TrB, WB_df



    ### Fuctions to be used in the model
    ### Fuctions to be used in the model
    def SAS_sampling(self, SWater_age = None, STrVol_age = None, Water_in_age = None, Water_out= None, InConc_age = None, alpha = None, Beta= None):
            if STrVol_age.size  == 0.0 or InConc_age.size == 0.0 : 
                STrVol_age= np.nan
                OutFlux=np.nan
                OutConc_age=np.nan
            else: ## Update temporary storage volume and concentration
                if  Water_in_age.size != 0:
                    SWater_age= np.roll(SWater_age,1)
                    SWater_age[0]=0.0
                    # Swater_age 0 to size of Water_in_age 
                    SWater_age[0:Water_in_age.size] = SWater_age[0:Water_in_age.size] + Water_in_age
                    SWater_age[SWater_age <= 1e-10]=0.0
                    SWater_age[SWater_age <= 0.0]= 0.0
                    STrVol_age=np.roll(STrVol_age,1)
                    STrVol_age[0]=0.0
                    np.nan_to_num(InConc_age , nan=0.0, posinf=0.0, neginf=0.0)
                    STrVol_age[0:Water_in_age.size] = STrVol_age [0:Water_in_age.size]+ Water_in_age*InConc_age
                    STrVol_age[SWater_age == 0.0]=0.0
                if np.nansum(SWater_age) > 1e-12 and np.nansum(Water_out) > 1e-12:
                    SWater_age_pdf=SWater_age/np.nansum(SWater_age)
                    SWater_age_cdf=np.nancumsum(SWater_age_pdf)
                    ii=0
                    while ii == 0: ##  Check if any Water out age is  be bigger then storage age
                        if alpha == 1 and Beta == 1:
                            SAS_pdf = SWater_age_pdf
                        else:
                            SAS_cdf=beta.cdf(SWater_age_cdf, a = alpha, b = Beta, loc = 0.0) ## Here define shape function for SAS
                            #Append CDF forst element to SAS_pdf as last element
                            SAS_pdf=np.append(SAS_cdf[0], np.diff(SAS_cdf))
                        Water_out_age = Water_out*SAS_pdf ##Now we get output AGe distribution and check here if  Water out age is be bigger then storage age
                        if any(Water_out_age > SWater_age + 0.005):  
                            if alpha < 1:
                                alpha=min(1,alpha + 0.01)
                            else:
                                Water_out_age = SWater_age
                                ii=1
                        else:
                            ii=1
                    if np.nansum(Water_out_age > SWater_age) != 0.0:
                        SAS_pdf=SWater_age_pdf
                        Water_out_age=Water_out*SAS_pdf
                    # Apend zero StrVOl_age where Water_out_age is zero
                    STrConc_age=STrVol_age/SWater_age
                    STrConc_age[SWater_age == 0.0]=0.0
                    OutFlux_age= Water_out_age*STrConc_age
                    OutFlux=np.nansum(OutFlux_age)
                    if Water_out == 0.0:
                        OutConc=0.0
                    else:
                        OutConc=OutFlux/Water_out

                    OutConc_age= STrConc_age
                    OutConc_age[Water_out_age == 0]=0.0
                    SWater_age=SWater_age - Water_out_age
                    SWater_age[SWater_age < 0.0]= 0.0
                    STrVol_age = STrVol_age - OutFlux_age
                    STrVol_age[SWater_age == 0.0]=0.0
                else:
                    Water_out_age= np.zeros(SWater_age.size)
                    OutFlux=0.0
                    OutConc=0.0
                    OutConc_age= np.zeros(SWater_age.size)
            return Water_out_age, OutConc_age, SWater_age, OutConc, STrVol_age, OutFlux



    ## Tracer Snow Storage

    def Tracer_sampling_snow(self, Prec, ConcP, RM, Tr_length, alpha, Beta, **kwargs):
            #Np.zeros to create matrix  nxn for  age distribution of water in the snow bucket
            rm_Age_dist = np.zeros(Tr_length+1, dtype=float64) # Rain melt age distribution at each time step
            rm_Conc_dist = np.zeros(Tr_length+1, dtype=float64) #Rain and melt concentration age distribution at each time step
            Flux_RM =   np.zeros(Tr_length+1, dtype=float64) #Total Flux of water out of the snow bucket
            RM_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) # RM age distribution matrix TTD
            Conc_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Concentration matrix RM
            RM_flux_MR = np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Total Flux of water out of the snow bucket matrix
            #Storage 
            ss_Age_dist = np.zeros(Tr_length+1, dtype=float64) #Stored water age distribution at each time step 
            ss_vol_dist = np.zeros(Tr_length+1, dtype=float64) #Stored water volume age distribution (Con*SS) at each time step
            ss_vol_dist_in = np.zeros(Tr_length+1, dtype=float64) #Stored water input volume age distribution (Con*SS) at each time step
            SS_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) # Storage age distribution matrix RTD
            SS_Flux_MR = np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Total Flux of water out of the snow bucket matrix
            # Lopp over the time series
            for i in range(Tr_length):
                #Rain and melt sampling
                rm_Age_dist, rm_Conc_dist, ss_Age_dist, rm_Conc, ss_vol_dist, rm_Flux = self.SAS_sampling(ss_Age_dist, ss_vol_dist_in, Prec[i], RM[i], ConcP[i], alpha, Beta)
                RM_Age_MR[:,i] = rm_Age_dist
                Flux_RM[i] = rm_Flux
                SS_Age_MR[:,i] = ss_Age_dist
                SS_Flux_MR[:,i] = ss_vol_dist
                RM_flux_MR[:,i] = rm_Age_dist*rm_Conc_dist
                Conc_Age_MR[:,i] = RM_flux_MR[:,i]/RM_Age_MR[:,i]
                #append zero to ConMR_Age where Rm_Age is zero
                Conc_Age_MR[RM_Age_MR[:,i] == 0.0,i] = 0.0
                if i < Tr_length:
                    ss_vol_dist_in = ss_vol_dist
                #if i is equal to Tr_lenght ss_vol_dist_in = ss_vol_dist    

            return RM_Age_MR, RM_flux_MR, Conc_Age_MR,  Flux_RM, SS_Age_MR, SS_Flux_MR

    #Tracer Interseption Storage 

    def Tracer_sampling_Si(self,RM_Age_MR,  Conc_Age_MR, pe, ei,  Tr_length, alpha, Beta, x):
            #Prepare the input and output data format 
            Pe_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #Output Age Distribution  each time step
            Pe_Conc_dist = np.zeros(Tr_length+1, dtype=float64) # interseptionout concantration distribution each time step
            Flux_Pe = np.zeros(Tr_length+1, dtype=float64) #Total Flux of water out of the interception bucket
            Si_Age_dist = np.zeros(Tr_length+1, dtype=float64) # interseption  storage age distribution each time step
            Si_vol_dist = np.zeros(Tr_length+1, dtype=float64) # Storage volume distribution each time step
            Si_vol_dist_in = np.zeros(Tr_length+1, dtype=float64) # Storage volume distribution as input to the next time step 
            Si_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) # interseption storage age distribution matrix
            Si_Flux_MR = np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Total Flux of water out of the interception bucket matrix
            Pe_Conc_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Concentration matrix effective Pe
            Pe_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Efective Pe age distribution matrix
            Pe_Flux_MR = np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Total Flux of pe
            # Evaporation fluxes 
            Ei_Age_dist = np.zeros(Tr_length+1,  dtype=float64) # EiOutput Age Distribution eachtime step
            Ei_Conc_dist = np.zeros(Tr_length+1, dtype=float64) # Evaporation concantration distribution for each time steps
            Ei_Age_MR =  np.zeros((Tr_length+1, Tr_length+1), dtype=float64)  #Ei Main Matrix age distribution 
            Ei_Flux_MR = np.zeros((Tr_length+1, Tr_length+1), dtype=float64) #Total Flux of water out of the snow bucket matrix
            Flux_Ei = np.zeros(Tr_length+1, dtype=float64) #Total Flux of water out of the interception bucket
            #Tracer balance interseption
            TB_si = np.zeros(Tr_length+1, dtype=float64)
            # Lopp over the time series
            for i in range(Tr_length):
                #Interseption sampling
                Pe_Age_dist, Pe_Conc_dist, Si_Age_dist, Pe_Conc, Si_vol_dist, Pe_Flux = self.SAS_sampling(Si_Age_dist, Si_vol_dist_in, RM_Age_MR[:,i], pe[i],  Conc_Age_MR[:,i],alpha, Beta)
                Ei_Age_dist, Ei_Conc_dist, Si_Age_dist, Ei_Conc, Si_vol_dist, Ei_Flux = self.SAS_sampling(Si_Age_dist, Si_vol_dist, x , ei[i], Conc_Age_MR[:,i], alpha, Beta)
                Pe_Age_MR[:,i] = Pe_Age_dist #This is actualy TTDS age distribution for out flux
                Pe_Flux_MR[:,i] = Pe_Age_dist*Pe_Conc_dist
                Flux_Pe[i] = Pe_Flux
                Pe_Conc_MR[:,i] = Pe_Conc_dist
                Ei_Age_MR[:,i] = Ei_Age_dist
                Ei_Flux_MR[:,i] = Ei_Conc_dist*Ei_Age_dist
                Flux_Ei[i] = Ei_Flux
                Si_Age_MR[:,i] = Si_Age_dist
                Si_Flux_MR[:,i] = Si_vol_dist
                #append zero to Conpe_Age where Rm_Age is zero
                Pe_Conc_MR[Pe_Age_MR[:,i] == 0.0,i] = 0.0
                TB_si[i] = np.nansum(RM_Age_MR[:,i]*Conc_Age_MR[:,i]) - Pe_Flux - Ei_Flux - np.nansum(Si_vol_dist) + np.nansum(Si_vol_dist_in) #Total water balance in the snow bucket
                if i < Tr_length:
                    Si_vol_dist_in = Si_vol_dist   

            return Pe_Age_MR, Pe_Flux_MR, Pe_Conc_MR, Ei_Age_MR, Ei_Flux_MR, Si_Age_MR, Si_Flux_MR,  Flux_Pe, Flux_Ei, TB_si


    ## Tracer Root zone 

    def Tracer_sampling_SR(self, Pe_Age_MR, Pe_Conc_MR, rtot, ea, rs, Tr_length, alpha, Beta, x, Srmax, ConSuAvg):
                ##loop for root zone balance 
                #fast responce rechage
                rf_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #fast responce Age Distribution  each time step
                rf_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #fast responce Concentration Distribution  each time step
                rf_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #fast responce Age Distribution TTD
                rf_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #fast responce output Concentration Distribution matrix for next input
                Flux_rf = np.zeros(Tr_length+1,  dtype=float64) #fast responce output Flux Distribution matrix for next input
                # Root zone Storage 
                Sr_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Age Distribution  each time step
                Sr_conc_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Concentration Distribution  each time step
                Sr_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Age Distribution RTD
                Sr_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Concentration Distribution matrix
                Sr_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Flux Distribution matrix
                Sr_vol_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Volume Distribution  each time step
                Sr_vol_dist_in = np.zeros(Tr_length+1,  dtype=float64) #Storage  Volume Distribution  each time step
            
                # ea
                ea_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #Evaporation  Age Distribution  each time step
                ea_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #Evaporation  Concentration Distribution  each time step
                ea_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Evaporation  Age Distribution TTD
                ea_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Evaporation  Concentration Distribution matrix
                Flux_ea = np.zeros(Tr_length+1,  dtype=float64) #Evaporation  Flux Distribution matrix for next input
                # rs
                rs_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #Recharge  Age Distribution  each time step
                rs_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #Recharge  Concentration Distribution  each time step
                rs_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Age Distribution TTD
                rs_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Concentration Distribution matrix
                Flux_rs = np.zeros(Tr_length+1,  dtype=float64) #Recharge  Flux Distribution matrix for next input

                Sr_Age_dist[0] = Srmax/2.0 ## initial condition
                Sr_vol_dist_in[0] =  ConSuAvg*(Srmax/2.0) ## initial condition
                #TRacer balance 
                TB_Sr = np.zeros(Tr_length+1,  dtype=float64) #Total water balance in the snow bucket
            
                # Lopp over the time series
                for i in range(Tr_length):
                    rf_Age_dist, rf_Conc_dist, Sr_Age_dist, rf_Conc, Sr_vol_dist, rf_Flux = self.SAS_sampling(Sr_Age_dist, Sr_vol_dist_in, Pe_Age_MR[:,i], rtot[i], Pe_Conc_MR[:,i], alpha[i], Beta)
                    ea_Age_dist, ea_Conc_dist, Sr_Age_dist, ea_Conc, Sr_vol_dist, ea_Flux = self.SAS_sampling(Sr_Age_dist, Sr_vol_dist, x , ea[i], Pe_Conc_MR[:,i], 1.0, Beta)
                    rs_Age_dist, rs_Conc_dist, Sr_Age_dist, rs_Conc, Sr_vol_dist, rs_Flux = self.SAS_sampling(Sr_Age_dist, Sr_vol_dist, x , rs[i], Pe_Conc_MR[:,i],  1.0, Beta)

                    rf_Age_MR[:,i] = rf_Age_dist
                    rf_Conc_MR[:,i] = rf_Conc_dist
                    Flux_rf[i] = rf_Flux
                    ea_Age_MR[:,i] = ea_Age_dist
                    ea_Conc_MR[:,i] = ea_Conc_dist
                    Flux_ea[i] = ea_Flux
                    rs_Age_MR[:,i] = rs_Age_dist
                    rs_Conc_MR[:,i] = rs_Conc_dist
                    Flux_rs[i] = rs_Flux
                    Sr_Age_MR[:,i] = Sr_Age_dist
                    Sr_Flux_MR[:,i] = Sr_vol_dist
                    Sr_Conc_MR[Sr_Age_MR[:,i] == 0.0,i] = 0.0
                    #Tracer balance 
                    TB_Sr[i] = np.nansum(Pe_Age_MR[:,i]*Pe_Conc_MR[:,i]) - (rf_Flux + ea_Flux + rs_Flux)  - np.nansum(Sr_vol_dist) + np.nansum(Sr_vol_dist_in)
                    
                    if i < Tr_length:
                        Sr_vol_dist_in = Sr_vol_dist

                return rf_Age_MR, rf_Conc_MR, Flux_rf, ea_Age_MR, ea_Conc_MR, Flux_ea, rs_Age_MR, rs_Conc_MR, Flux_rs, Sr_Age_MR, Sr_Flux_MR, TB_Sr



    ## Tracer  preferential flow divitions to fast responce, recharge to groundwater and Overland flow
    def Rf_division(self, rf_Age_MR, rf_Conc_MR, Tr_length, cp, cn ):
        ## Divtions Here  for fast and overland flows
        ## rfs fast responce to slow responce
        rfs_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Age Distribution TTD
        rfs_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Concentration Distribution matrix
        rfs_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        Flux_rfs = np.zeros((Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        #rfn  fast responce 
        rfn_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Age Distribution TTD
        rfn_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Concentration Distribution matrix
        rfn_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        Flux_rfn = np.zeros((Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        #rff propotion of fast responce going to  fast responce
        rff_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Age Distribution TTD
        rff_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Concentration Distribution matrix
        rff_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        Flux_rff = np.zeros((Tr_length+1),  dtype=float64) #Recharge  Flux Distribution matrix
        #Qo propotion of fast responce going tostream directly 
        Qo_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Overland flow  Concentration Distribution matrix
        Qo_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Overland f low  Age Distribution TTD
        Qo_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Overland flow  Flux Distribution matrix
        Flux_Qo = np.zeros((Tr_length+1),  dtype=float64) #Overland flow  Flux Distribution matrix
        
        # first divition of fast responce
        rfs_Age_MR = (1-cp)*rf_Age_MR
        rfs_Conc_MR = rf_Conc_MR
        rfs_Flux_MR = rfs_Age_MR*rfs_Conc_MR
        Flux_rfs = np.nansum(rfs_Flux_MR, axis=0)
        rff_Age_MR = cp*rf_Age_MR
        rff_Conc_MR = rf_Conc_MR
        rff_Flux_MR = rff_Age_MR*rff_Conc_MR
        Flux_rff = np.nansum(rff_Flux_MR, axis=0)
        # Second divition of fast responce 
        Qo_Age_MR = cn*rff_Age_MR
        Qo_Conc_MR = rff_Conc_MR
        Qo_Flux_MR=Qo_Age_MR*Qo_Conc_MR
        Flux_Qo = np.nansum(Qo_Flux_MR, axis=0)
        rfn_Age_MR = (1-cn)*rff_Age_MR
        rfn_Conc_MR = rff_Conc_MR
        rfn_Flux_MR = rfn_Age_MR*rfn_Conc_MR
        Flux_rfn = np.nansum(rfn_Flux_MR, axis=0)

        return rfs_Age_MR, rfs_Conc_MR, rfs_Flux_MR, Flux_rfs, rfn_Age_MR, rfn_Conc_MR, rfn_Flux_MR, Flux_rfn, Qo_Age_MR, Qo_Conc_MR, Qo_Flux_MR, Flux_Qo

    #Tracer fast responce

    def Tracer_sampling_Sf(self, rfn_Age_MR, rfn_Conc_MR, Qof, Qf, Tr_length, alpha, Beta, x, Sfmax):
            Qf_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #fast responce Age Distribution  each time step
            Qf_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #fast responce Concentration Distribution  each time step
            Qf_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #fast responce Age Distribution TTD
            Flux_Qf = np.zeros((Tr_length+1),  dtype=float64) #fast responce Flux Distribution matrix
            #QOF 
            Qof_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #overland flow Age Distribution  each time step
            Qof_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #overland flow Concentration Distribution  each time step
            Qof_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #overland flow Age Distribution TTD
            Flux_Qof = np.zeros((Tr_length+1),  dtype=float64) #overland flow Flux Distribution matrix
            #ET
            #Storage 
            Sf_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Age Distribution  each time step
            Sf_Age_dist[0] = Sfmax/2
            Sf_conc_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Concentration Distribution  each time step
            Sf_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Age Distribution RTD
            Sf_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Concentration Distribution matrix
            Sf_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #Storage  Flux Distribution matrix
            Sf_vol_dist = np.zeros(Tr_length+1,  dtype=float64) #Storage  Volume Distribution  each time step
            Sf_vol_dist_in = np.zeros(Tr_length+1,  dtype=float64) #Storage  Volume Distribution  each time step
            Sf_vol_dist_in[0] = 0.0 ## initial condition

            TB_sf = np.zeros(Tr_length+1,  dtype=float64) #Storage  Age Distribution  each time step
            # Lopp over the time series
            for i in range(Tr_length):
                Qof_Age_dist,  Qof_Conc_dist, Sf_Age_dist,  Qof_Conc, Sf_vol_dist, Qof_Flux = self.SAS_sampling(Sf_Age_dist, Sf_vol_dist_in, rfn_Age_MR[:,i], Qof[i], rfn_Conc_MR[:,i],alpha, Beta)
                Qf_Age_dist, Qf_Conc_dist, Sf_Age_dist, Qf_Conc, Sf_vol_dist, Qf_Flux = self.SAS_sampling(Sf_Age_dist, Sf_vol_dist, x , Qf[i], rfn_Conc_MR[:,i], alpha, Beta)

                Qof_Age_MR[:,i] = Qof_Age_dist
                Flux_Qof[i] = Qof_Flux
                Qf_Age_MR[:,i] = Qf_Age_dist
                Flux_Qf[i] = Qf_Flux

                Sf_Age_MR[:,i] = Sf_Age_dist
                Sf_Conc_MR[:,i] = Sf_vol_dist/Sf_Age_dist
                Sf_Flux_MR[:,i] = Sf_vol_dist
                Sf_Conc_MR[Sf_Age_MR[:,i] == 0.0,i] = 0.0

                TB_sf[i] = np.nansum(rfn_Age_MR[:,i]*rfn_Conc_MR[:,i]) - (Qf_Flux + Qof_Flux ) - np.nansum(Sf_vol_dist)+np.nansum(Sf_vol_dist_in)
                if i < Tr_length-1:
                    Sf_vol_dist_in = Sf_vol_dist
            return Qf_Age_MR, Qof_Age_MR,  Sf_Age_MR, Sf_Conc_MR, Sf_Flux_MR, Flux_Qf, Flux_Qof, TB_sf

    ## Tracer Slow responce 
    def Tracer_sampling_SG(self, Qin_Age_MR, Conc_in_MR, Qstot, Ql, Tr_length, alpha, Beta, x,  SSp, ConAvg):
            #outs 
            Qstot_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #slow responce Age Distribution  each time step
            Qstot_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #slow responce Concentration Distribution  each time step
            Qstot_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Age Distribution TTD
            Qstot_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Concentration Distribution matrix
            Flux_Qstot = np.zeros((Tr_length+1),  dtype=float64) #slow responce Flux Distribution matrix
            #Loses
            Ql_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #slow responce Age Distribution  each time step
            Ql_Conc_dist = np.zeros(Tr_length+1,  dtype=float64) #slow responce Concentration Distribution  each time step
            Ql_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Age Distribution TTD
            Ql_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Age Distribution TTD
            Ql_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Concentration Distribution matrix
            Flux_Ql = np.zeros((Tr_length+1),  dtype=float64) #slow responce Flux Distribution matrix
            ## SLow Responcew Flux
            SSa_Age_dist = np.zeros(Tr_length+1,  dtype=float64) #slow responce Age Distribution  each time step
            SSa_Age_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Age Distribution TTD
            SSa_Conc_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Concentration Distribution matrix
            SSa_Flux_MR = np.zeros((Tr_length+1,Tr_length+1),  dtype=float64) #slow responce Flux Distribution matrix
            SSa_vol_dist_in = np.zeros(Tr_length+1,  dtype=float64) #slow responce Volume Distribution  each time step
            TB_Ss = np.zeros(Tr_length+1,  dtype=float64) #slow responce tracer balance 
            SSa_Age_dist[0] =  SSp ## initial condition
            SSa_vol_dist_in[0] = SSa_Age_dist[0]*ConAvg ## initial condition for input
            for i in range(Tr_length):
                Qstot_Age_dist,  Qstot_Conc_dist, SSa_Age_dist,  Qstot_Conc, SSa_vol_dist, Qstot_Flux = self.SAS_sampling(SSa_Age_dist, SSa_vol_dist_in,  Qin_Age_MR[:,i], Qstot[i], Conc_in_MR[:,i], alpha, Beta)
                Ql_Age_dist, Ql_Conc_dist, SSa_Age_dist,  Ql_Conc, SSa_vol_dist, Ql_Flux = self.SAS_sampling(SSa_Age_dist, SSa_vol_dist, x, Ql[i], Conc_in_MR[:,i], alpha, Beta)
                Qstot_Age_MR[:,i] = Qstot_Age_dist
                Qstot_Conc_MR[:,i] = Qstot_Conc_dist
                Flux_Qstot[i] = Qstot_Flux
                Ql_Age_MR[:,i] = Ql_Age_dist
                Ql_Conc_MR[:,i] = Ql_Conc_dist
                Ql_Flux_MR[:,i] = Ql_Flux
                Flux_Ql[i] = Ql_Flux
                SSa_Age_MR[:,i] = SSa_Age_dist
                SSa_Flux_MR[:,i] = SSa_vol_dist
                SSa_Conc_MR[:,i] = SSa_vol_dist/SSa_Age_dist
                SSa_Conc_MR[SSa_Age_MR[:,i] == 0.0,i] = 0.0
                TB_Ss[i] = np.nansum(Qin_Age_MR[:,i]*Conc_in_MR[:,i]) - Qstot_Flux - Ql_Flux- np.nansum(SSa_vol_dist)+ np.nansum(SSa_vol_dist_in)
                if i < Tr_length-1:
                    SSa_vol_dist_in = SSa_vol_dist
            return Qstot_Age_MR, Flux_Qstot, SSa_Age_MR, SSa_Conc_MR, SSa_Flux_MR, TB_Ss, Qstot_Conc_MR, Ql_Age_MR, Ql_Conc_MR, Ql_Flux_MR, Flux_Ql

    ## Sampling Functions for Tracer
    
    @staticmethod
    @njit
    def get_overland_flow(pr, rf, Ptresh = 5.0,  cn = 0.4):
        n = pr.size
        # Create empty arrays to store the fluxes and states
        Qo = zeros(n, dtype=float64)  # Overland flow
        rfn = zeros(n, dtype=float64)  # fas recharge
        Cn  = zeros(n, dtype=float64)
        # Fast responce bucket
        for t in range(n):
            # Make sure the solution is larger then 0.0 and smaller than sf
            if pr[t] > Ptresh:
                Cn[t] = cn
                # if (sr[t]/srmax)> 0.7:
                # Cn[t]= max(0.0, exp(- srmax/sr[t]))
                # Cn[t] = cn*(1- Ptresh/pr[t])
                # else:
                #   Cn[t] = cn
                Qo[t]  = max(rf[t]*Cn[t], 0.0)  # overland flow occur
                rfn[t]  = (1-Cn[t])*rf[t]
            else:
                Qo[t]= 0.0
                rfn[t]  = rf[t]
        return Qo, rfn, Cn
    
    @staticmethod
    @njit
    def get_slow_res_balance(rs, Ka=0.05, Kp = 0.02, SSp = 5000.0, dt=1.0):
        n = rs.size
        # Create empty arrays to store the fluxes and states
        ssa = zeros(n + 1, dtype=float64)  # Slow Storage
        ssp = zeros(n + 1, dtype=float64)  # Slow Storage passive component
        ssan = zeros(n + 1, dtype=float64)  # Slow Storage
        Stot_out = zeros(n, dtype=float64)  # Slow Storage
        prop_QsQl = zeros(n, dtype=float64)  # Slow Storage
        Qs = zeros(n, dtype=float64)  # slow runoff Flux
        Qst = zeros(n, dtype=float64)  #  slow total runoff Flux
        Ql = zeros(n, dtype=float64)  # slow loss runoff Flux
        ssp[0] = SSp  # Set the initial system state to half-full
        # Make sure the solution is larger then 0.0 and smaller than ssa
        # GW bucket 
        for t in range(n):
          ssan[t] =  ssa[t] + rs[t] + ssp[t]
          # Calculate the slow runoff Flux
          Stot_out[t]=((Ka*ssp[t])/(Ka+Kp))+(ssan[t]*exp((-Ka-Kp)*dt))-((Ka*ssp[t]*exp((-Ka-Kp)*dt))/(Ka+Kp))
          Qst[t] = ssan[t] - Stot_out[t] ## total of loss and Qs slow responce
          prop_QsQl[t]=max(0,((Ka*ssan[t])-(Ka*ssp[0]))/(Kp*ssan[t]))
          Qs[t] =prop_QsQl[t]*Qst[t]/(prop_QsQl[t]+1)
          Ql[t] = Qst[t]/(prop_QsQl[t]+1)
          if Stot_out[t]>=ssp[0]:
            ssa[t + 1]=Stot_out[t]-ssp[0]
            ssp[t+1]=ssp[0]
          else:
             ssa[t + 1]=0.0
             ssp[t+1]=Stot_out[t]
        return ssa[:-1], Qs, Ql, ssp[:-1]

    @staticmethod
    @njit
    def get_fast_res_balance(rfn, sfmax=20.0, dt=1.0, Bf = 1e-5, kf=0.02):
        n = rfn.size
        # Create empty arrays to store the fluxes and states
        sf = zeros(n + 1, dtype=float64)  # Fast Storage
        sfn = zeros(n + 1, dtype=float64)  # Fast Storage
        sf[0] = 0.5 * sfmax  # Set the initial system state to half-full
        Qof = zeros(n, dtype=float64)  # overland flow Flux
        Qf = zeros(n, dtype=float64)  # fast runoff Flux
        # Fast responce bucket
        for t in range(n):
            # Make sure the solution is larger then 0.0 and smaller than sf
            sfn[t] = sf[t]+ rfn[t]
            if sfn[t] > sfmax:
                cof = min((sfn[t]/sfmax)**Bf, 1.0)
                Qof[t]  = max((sfn[t]*cof -sfmax), 0.0)  # overland flow occur
                sfn[t] = sfn[t] -Qof[t]
                Qf[t] = max(0.0, sfn[t]*(1.0-exp(-kf*dt))) ## storage will be in full capacity 
            else:
                Qof[t]= 0.0
                Qf[t] = max(0.0, sfn[t]*(1.0-exp(-kf*dt)))
            # Calculate the fast runoff Flux

            # Update fast storage in the root zone

            sf[t + 1] = max(0.0, (sf[t] +rfn[t]- Qf[t] - Qof[t]))
            
        return sf[:-1], Qf, Qof

    @staticmethod
    @njit
    def get_root_zone_balance(pr, pe, ep, srmax=250.0, lp=0.25, Rsmax=2.0, beta=2.0, dt=1.0, cp = 0.8, alpha_Su = 1.0, Ptresh= 10):
        n = pe.size
        # Create empty arrays to store the fluxes and states
        sr = zeros(n + 1, dtype=float64)  # Root Zone Storage State
        sr[0] = 0.5 * srmax  # Set the initial system state to half-full
        srn = zeros(n + 1, dtype=float64)  # Root Zone Storage State
        ea = zeros(n, dtype=float64)  # Actual evaporation Flux
        rs = zeros(n, dtype=float64)  # Recharge Flux slow
        rfs = zeros(n, dtype=float64) # Recharge Flux from fast to slow
        rff = zeros(n, dtype=float64)  # Fast recharge Flux
        rtot = zeros(n, dtype=float64)  # Fast recharge Flux
        cr = zeros(n, dtype=float64)  # CR
        capTemp = zeros(n, dtype=float64) ## Capacity for overflow
        alphat = zeros(n, dtype=float64) ## Capacity for overflow
        for t in range(n):
            # Make sure the solution is larger then 0.0 and smaller than sr
            srn[t] = max((sr[t]+ pe[t]), 0.0)
            cap = ((1+beta)*srmax)*(1-(max(0,((1-sr[t]/srmax))**(1/(1+beta)))))
            #check if alpha su is nan if its not funtion Pe/srmax  if it is not not nan funtion of Sr/srmax  
            if np.isnan(alpha_Su) == True:
                alphat[t] = max(1e-09, (1-((srn[t])/srmax)))
            elif pr[t] >= Ptresh:
                alphat[t] = max(1e-09, alpha_Su) #If there is rain and the soil is not saturated, then alpha su o.1
            else:
                alphat[t] =  max(1e-09, (1-((srn[t]/srmax)*(1-alpha_Su))))  
            if pe[t] > 0.0:
                if (pe[t]+cap)>=(1+beta)*srmax:
                  capTemp[t]=pe[t]-srmax +sr[t] # Fast recharge
                else:
                  capTemp[t]= pe[t] - srmax + sr[t]+srmax*((1-(pe[t]+cap)/((1+beta)*srmax))**(1+beta))

                cr[t] = min(1,capTemp[t]/pe[t])
            else:
              cr[t] = 0.0
            rtot[t] = cr[t]*pe[t] ## to fast recharge
            srn[t] = srn[t] - rtot[t] # update the storage for evaporation
            if srn[t] > srmax:
                rtot[t] = rtot[t] + srn[t] - srmax  # Surface runoff
                srn[t] = srmax
            elif srn[t] < 0.0:
                srn[t] = 0.0
            else:
                 srn[t] = srn[t]
            rff[t] =  cp*rtot[t] # Fast recharge becareful with cr<1 in this case
            rfs[t] = (1-cp)*rtot[t] # Fast recharge to slow recharge
            # # Calculate evaporation from the root zone reservoir
            # if temp[t] > ttr and pmean[t] > 3.0:
            #     ea[t]  = kp*ep[t]*min(1, srn[t]/lp)
            # else:
            ea[t] = ep[t]*min(1, (srn[t]/srmax)*1/lp)
            # Calculate the recharge flux and update it
            #ea[t] = max(ep[t], ea[t])
            srn[t] = max(0.0, (srn[t] - ea[t]))
            rs[t] = min(Rsmax*srn[t]/srmax, srn[t])
            #rs[t] = min(Rsmax*1/(1+exp(((-srn[t]/srmax)+0.5)/beta)), srn[t])
            # Update storage in the root zone
            sr[t + 1] = srn[t] - min(rs[t], srn[t]) #3 update the strage for the next time step

        return sr[:-1], rs, ea, rff, rfs, pe, cr,  alphat
    @staticmethod
    @njit 
    def get_interception_balance(pr, ep, Imax=2.0, dt=1.0):
        n = pr.size
        si = zeros(n + 1, dtype=float64)  # Interception Storage State
        pe = zeros(n+ 1, dtype=float64)  # Effective precipitation Flux
        sm =  zeros(n, dtype=float64)  # Effective precipitation Flux
        ei = zeros(n, dtype=float64)  # Interception evaporation Flux

        for t in range(n):
            # Interception bucket
            sm[t] = max((si[t]+ pr[t]), 0.0)
            pe[t] = max((sm[t] - Imax), 0.0)
            ei[t] = min(ep[t], (sm[t]-pe[t]))
            si[t + 1] =  max((sm[t] - pe[t] - ei[t]), 0.0)

        return si[:-1], ei, pe[:-1]
    @staticmethod
    @njit
    def get_snow_balance(prec, temp, tt=0.0, fmelt=2.0):
        n = prec.size
        # Create empty arrays to store the fluxes and states
        ss = zeros(n + 1, dtype=float64)  # Snow Storage
        ssn = zeros(n + 1, dtype=float64)
        ps = where(temp <= tt, prec, 0.0)  # Snowfall
        m = where(temp > tt, fmelt * (temp - tt), 0.0)  # Potential Snowmelt
        # Snow bucket
        for t in range(n):
            ssn[t] = ss[t] + ps[t]
            if temp[t] > tt:
                m[t] = min(m[t],  ssn[t])
            ss[t + 1] = (ssn[t] - m[t])

        return ss[:-1], ps, m
    
    def get_tracer_balance(self, prec, evap,  temp, ConP, ConAvg,  ConSuAvg, p, Case,  **kwargs):
    
        Age_MR, Flux_df, Storage_Age_MR, Storage_Flux_MR, TrB, WB_df = self.Tracer_Tracking(prec=prec, evap=evap, temp=temp, ConP =ConP, ConAvg = ConAvg,  ConSuAvg = ConSuAvg,  p=p, **kwargs)
        #If the case is "CAL" calibration return Flux_df, Age_MR and WB_df
        if Case == 'Cal':
            del Storage_Flux_MR, Storage_Age_MR, TrB
            return Flux_df, Age_MR , WB_df

        else :
    
            return  Age_MR, Flux_df, Storage_Age_MR, Storage_Flux_MR, TrB, WB_df

