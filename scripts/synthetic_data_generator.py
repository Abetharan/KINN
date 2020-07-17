import numpy as np 
import itertools
import h5py
import os 
import matplotlib.pyplot as plt 
import random as rand 
import math
import pandas as pd
from scipy.special import comb

rand.seed(0)
np.random.seed(0)
#global constant
kb = 1.38064852e-23
protonMass = 1.6726219e-27
electronMass = 9.10938356e-31
epsilon_0 = 8.85418782e-12
electronCharge = 1.60217662e-19
hbar = 1.054571800e-34
c = 299792458
ev_converter = electronCharge/kb
def TwoTemperatureBath(nx, L, Z, Te_top, Te_low):

    #Two bath problem
    coord = np.linspace(0, L, nx + 1)
    central = np.array([(coord[i + 1] + coord[i]) / 2 for i in range(len(coord) - 1)])
    gammaFactor = 5/3
    Tu = Te_top 
    Td = Te_low 
        # T(I) = (T_UP ** (7.00D00/2.00D00) + X_GRID(I)/X_GRID(NUM_X) * &
        #   (T_DIV ** (7.00D00/2.00D00) - T_UP ** (7.00D00/2.00D00))) ** (2.00D00/7.00D00)
        #   n_TOT(I) = DENS_UP * T_UP/T(I)
    Te = (Tu**3.5 + (central/ central[-1]) * (Td**3.5 - Tu**3.5))**(1/3.5)
    Te[0] = Tu
    Te[-1] = Td
    Ti = Te
    nu = 8.1E21
    ne = nu * (Tu/Te)
    ni = ne / Z 
    Z = np.zeros(nx) + Z
    density = ni * 1.67E-27
    return(central,Te, ne,Z)

def epperlein_short(nx, L, Z_ = 37.25, ne_ = 1e27, Te_ = 100., perturb = 1e-3, sin = False):

    x_l = 0
    x_u = L
    initial_coord = np.linspace(x_l, x_u, nx+1, dtype=np.float64)
    x_centered = np.array([(initial_coord[i] + initial_coord[i+1]) /2 for i in range(len(initial_coord) -1)], dtype=np.float64)
    Z = np.zeros(nx) + Z_#37.25
    
    ne = np.zeros(nx) + ne_# 1e27
    #temperatureE = np.linspace(3, 300, nx)
    
    ##Relevant for periodic systems .. fluid cannot be
    if sin: 
        Te = np.zeros(nx, dtype=np.float64) + Te_  + perturb * Te_  * np.sin((2*np.pi * x_centered) / np.max(x_centered), dtype=np.float64)
    else:
    ##Reflective
        Te = np.zeros(nx, dtype=np.float64) + Te_  + perturb * Te_  * np.cos((np.pi * x_centered) / np.max(x_centered), dtype=np.float64)


    return(x_centered, Te, ne, Z)

def AllSmoothRamp(nx, L, lower_cutof, upper_cutof, Te_begin, Te_end, Te_low, Te_top, ne_begin, ne_end, ne_low, ne_top, Z_begin, Z_end, z_low, z_top):
    x_up = np.linspace(0, lower_cutof*L, int(nx*0.33333))
    step = x_up[1] - x_up[0]
    x_ramp = np.linspace(lower_cutof*L + step/10, upper_cutof*L, int(nx*0.33333) + 1)
    step = x_ramp[1] - x_ramp[0]
    x_down = np.linspace(upper_cutof*L+step/10, L, int(0.33333*nx))
    x = np.concatenate((x_up, x_ramp, x_down))
    # x = np.linspace(0, L, nx + 1)
    #x = np.linspace(0, L, nx + 1)
    x_centered = np.array([(x[i] + x[i + 1]) /2 for i in range(len(x) -  1)])
    
    def smooth(x, lower_value, upper_value, lower_limit, upper_limit):
        Var = np.zeros(len(x))
        counter = 0
        for i in range(len(x)):
            if x[i] < lower_limit:
                Var[i] =  lower_value
            elif x[i] > upper_limit:
                Var[i] = upper_value
            else:
                # Var[i] = (upper_value - lower_value) * np.tanh((x[i] - lower_limit) / (upper_limit - lower_limit)) + lower_value
                # Var[i] = (upper_value - lower_value) * (1 /(1 + np.exp((x[i] - lower_limit) / (upper_limit - lower_limit))))+ lower_value
            #     # Var[i] = (upper_value - lower_value) * (3 * ((x[i] - lower_limit) / (upper_limit - lower_limit))**2 
                # - 2 * ((x[i] - lower_limit) / (upper_limit - lower_limit))**3) + lower_value
                Var[i] = (upper_value - lower_value) * (6 * ((x[i] - lower_limit) / (upper_limit - lower_limit))**5 
                - 15 * ((x[i] - lower_limit) / (upper_limit - lower_limit))**4 + 10 * ((x[i] - lower_limit) / (upper_limit - lower_limit))**3) + lower_value
                # print(Var[i])
        return Var         

    Te_Up = Te_top 
    Te_Down = Te_low 
    lower_limit = lower_cutof*L#x_centered[int(nx*0.4)]
    indices_lower_limt = np.where(x <= lower_limit)
    upper_limit = upper_cutof*L#x_centered[int(nx*0.7)]
    indices_upper_limt = np.where(x >= upper_limit)
    indices_upper_limt = indices_upper_limt[0] - 1

    Te = smooth(x_centered, Te_Up, Te_Down, lower_limit, upper_limit)
    lower_m = ((Te_Up +1- Te_begin) / (x_centered[indices_lower_limt[0][-1]] - x_centered[indices_lower_limt[0][0]]))
    lower_c = Te_begin
    Te[indices_lower_limt[0]] =  lower_m* x_centered[indices_lower_limt[0]] + lower_c
    upper_m = ((Te_end- Te_Down) / (x_centered[indices_upper_limt[-1]] - x_centered[indices_upper_limt[0]]))
    upper_c = Te[indices_upper_limt[0]] - upper_m*x_centered[indices_upper_limt[0]]
    Te[indices_upper_limt] =  upper_m * x_centered[indices_upper_limt] + upper_c
    
    ne_Up = ne_top * 1e26
    ne_Down = ne_low * 1e26
    ne = smooth(x_centered, ne_Up, ne_Down, lower_limit, upper_limit) 
    lower_m = ((ne_Up +1- ne_begin) / (x_centered[indices_lower_limt[0][-1]] - x_centered[indices_lower_limt[0][0]]))
    lower_c = ne_begin
    ne[indices_lower_limt[0]] =  lower_m* x_centered[indices_lower_limt[0]] + lower_c
    upper_m = ((ne_end- ne_Down) / (x_centered[indices_upper_limt[-1]] - x_centered[indices_upper_limt[0]]))
    upper_c = ne[indices_upper_limt[0]] - upper_m*x_centered[indices_upper_limt[0]]
    ne[indices_upper_limt] =  upper_m * x_centered[indices_upper_limt] + upper_c

    Z_Up = z_top#2
    Z_Down = z_low#36.64
    Z = smooth(x_centered, Z_Up, Z_Down, lower_limit, upper_limit) 
    lower_m = ((Z_Up +1- Z_begin) / (x_centered[indices_lower_limt[0][-1]] - x_centered[indices_lower_limt[0][0]]))
    lower_c = Z_begin
    Z[indices_lower_limt[0]] =  lower_m* x_centered[indices_lower_limt[0]] + lower_c
    upper_m = ((Z_end- Z_Down) / (x_centered[indices_upper_limt[-1]] - x_centered[indices_upper_limt[0]]))
    upper_c = Z[indices_upper_limt[0]] - upper_m*x_centered[indices_upper_limt[0]]
    Z[indices_upper_limt] =  upper_m * x_centered[indices_upper_limt] + upper_c
    
    plt.figure(1)
    plt.plot(x_centered, Te , 'xr-', label = 'Te')
    plt.legend()
    plt.figure(2)
    plt.plot(x_centered, ne,'xr-', label = 'ne')
    plt.legend()
    plt.figure(3)
    plt.plot(x_centered, Z,'xr-', label = 'Z')
    plt.legend()
    plt.figure(4)
    plt.show()
    return(x_centered, Te, ne, Z)


def smoothstep(x, x_min=0, x_max=1, N=1):
    #https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result
def Ramp(nx, L, lower_cut_off, upper_cut_off, Te_low, Te_up, ne_low, ne_up, Z_low, Z_up):
    x = np.linspace(0, L, nx + 1)
    x_centered = np.array([(x[i] + x[i + 1]) /2 for i in range(len(x) -  1)])
    lower_limit = lower_cut_off*L#x_centered[int(nx*0.4)]
    upper_limit = upper_cut_off*L#x_centered[int(nx*0.7)]

    ramp = smoothstep(x_centered, lower_limit, upper_limit, N = 2)
    Te = (ramp * (Te_up - Te_low)) + Te_low

    ne = np.flip((ramp * (ne_up - ne_low)) + ne_low)

    
    Z = np.flip((ramp * (Z_up - Z_low)) + Z_low)
    # plt.figure(0)
    # plt.plot(x_centered, Te, label = 'Te')
    # plt.figure(1)
    # plt.plot(x_centered, ne, label  = 'ne')
    # plt.figure(2)
    # plt.plot(x_centered, Z, label = 'Z')
    
    # plt.show()
    
    return(x_centered, Te, ne, Z)

def SpitzerHarm(x, T, ne, Z):
    
    def lambda_ei(T_norm , n_norm, Z_norm, return_arg = False):
        coulomb_logs = []
        T_norm = T_norm

        for T,n,Z in zip(T_norm, n_norm, Z_norm):
            if T < 10.00 * Z ** 2:
                result = 23.00 - math.log(math.sqrt(n * 1.00E-6) * Z * (T) ** (-3.00/2.00))
            else:
                result = 24.00 - math.log(math.sqrt(n * 1.00E-6) / (T))   

            if return_arg:
                return result
            else:
                coulomb_logs.append(result)
        return coulomb_logs

    coulomb_log = np.array(lambda_ei(T, ne , Z))
    kappaE = 1.843076667547614E-10 * pow(T, 2.5) * pow(coulomb_log, -1) *  pow(Z, -1)
    nx = len(Te)
    HeatFlowE = np.zeros(nx + 1)
    gradTe = np.zeros(nx + 1)
    for i in range(1, nx):
        centered_ke = 0.5 * (kappaE[i] + kappaE[i - 1])
        gradTe[i] = ((Te[i] - Te[i - 1]) / (x[i] - x[i - 1]))
        HeatFlowE[i] = centered_ke * gradTe[i] 
    
    HeatFlowE[0] = 0
    HeatFlowE[-1] = 0
    return(-1 * HeatFlowE[1:-1], gradTe[1:-1])

def santityCheck(array):

    if any(array < 0):
        return 1

    for i in range(2, len(array) - 2):
        mean = np.mean(array[int(i - 2): int(i + 2)])
        if array[i] > 1.2*mean or  array[i] < mean*0.8:
            return 1 
    
    return 0

def getTwoRandom():
    k = rand.random()
    k_1 = rand.random()
    if k > k_1:
        return getTwoRandom() 
    else:
        return(k, k_1)
def getBiggerOther(array, n):
    k = array[np.random.randint(0, n - 1)]
    k_1 = array[np.random.randint(0, n - 1)]
    if k > k_1:
        return getBiggerOther(array, n)
    
    return(k, k_1)

def saveToHDF5(dset, data):
    dset.create_dataset('data', data = data,)

def permutate(Te, ne, Z):
    Te_perms = [Te, np.flip(Te)]
    ne_perms = [ne, np.flip(ne)]
    Z_perms = [Z, np.flip(Z)]
    arrays = [Te_perms, ne_perms, Z_perms]
    return list(itertools.product(*arrays))

no_samples = 5000
split = 4000
base_path = "/Users/shiki/DATA/KINN_RELATED"
Data_h5 = h5py.File(os.path.join(base_path, "Data.hdf5"), "w")
trgradte = Data_h5.create_group("GradTe")
trTe = Data_h5.create_group("Te")
trne = Data_h5.create_group("ne")
trZ = Data_h5.create_group("Z")
trqe = Data_h5.create_group("qe")
trx = Data_h5.create_group("x")

ne_range = np.linspace(1e19, 1e28, 1000) #1e21-1e28
te_range =  np.linspace(1, 3000, 1000)# 1ev - 3kev
z_range =  np.linspace(1, 50, 1000)#2 - 38
L_range =  np.linspace(3E-3, 3, 1000)#3mm - 3m
nx = 100
n = 1000
j = 0
cell_wall_df = pd.DataFrame(columns=['GradT', 'qe'])
cell_centre_df = pd.DataFrame(columns=['Te','ne', 'Z'])
df_grad_te = np.array([])
df_qe = np.array([])
df_ne = np.array([])
df_x = np.array([])
df_Z = np.array([])
df_Te = np.array([])

def appendAllPermutations(x_arr, perms, df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x):
    for _, val in enumerate(perms):
        tdf_t = np.array(val[0])
        tdf_ne = np.array(val[1])
        tdf_Z = np.array(val[2])
        qe, gradTe = SpitzerHarm(x, val[0], val[1], val[2])
        tdf_qe = np.array(qe)
        tdf_gradT = np.array(gradTe)
        tdf_x = np.array(x_arr)
        
        df_Te = np.concatenate((df_Te, tdf_t))
        df_ne = np.concatenate((df_ne, tdf_ne))
        df_qe = np.concatenate((df_qe, tdf_qe))
        df_grad_te = np.concatenate((df_grad_te, tdf_gradT))
        df_Z = np.concatenate((df_Z, tdf_Z))
        df_x = np.concatenate((df_x, tdf_x))
    
    return  df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x

        

for i in range(no_samples + 1):
    L = L_range[np.random.randint(0, n)]
    z_1, z_2 = getBiggerOther(z_range, n)
    t_1, t_2 = getBiggerOther(te_range, n)
    n_1, n_2 = getBiggerOther(ne_range, n)
    lower, upper = getTwoRandom()  
    Te_random = np.random.uniform(1E-6, 0.9)
    # ne_random = np.random.uniform(0.90, 1.1)
    
    x, Te, ne, Z = epperlein_short(nx, L ,z_1, n_1, t_1, Te_random) 
    if bool(santityCheck(Te)):
        continue
    if bool(santityCheck(ne)):
        continue
    if bool(santityCheck(Z)):
        continue 

    all_permutations_epp = permutate(Te, ne, Z)
    df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x = appendAllPermutations(x, all_permutations_epp,df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x)

    x, Te, ne, Z = TwoTemperatureBath(nx, L, z_1, t_1, t_2) 
    if bool(santityCheck(Te)):
        continue
    if bool(santityCheck(ne)):
        continue
    if bool(santityCheck(Z)):
        continue 

    all_permutations_ttbath = permutate(Te, ne, Z)
    df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x = appendAllPermutations(x, all_permutations_ttbath,df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x)    
    x, Te, ne, Z = Ramp(nx, L, lower_cut_off = lower , upper_cut_off = upper, Te_low = t_1, Te_up = t_2, 
                        ne_low = t_1, ne_up = t_2 , Z_low = z_1, Z_up = z_2) 
    if bool(santityCheck(Te)):
        continue
    if bool(santityCheck(ne)):
        continue
    if bool(santityCheck(Z)):
        continue 
    
    all_permutations_ramp = permutate(Te, ne, Z)
    df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x = appendAllPermutations(x, all_permutations_ramp,df_Te, df_ne, df_qe, df_grad_te, df_Z, df_x)


    # x, Te, ne, Z = AllSmoothRamp(nx, L, lower_cutof = lower , upper_cutof = upper, Te_begin = t_2 * upper, 
                                # Te_end = t_1 * lower, Te_low = t_1, Te_top = t_2, ne_begin =n_2 * upper,
                                # ne_end =n_1*lower, ne_low = n_1, ne_top =n_2 , Z_begin = z_2*upper,
                                # Z_end = z_1*lower ,z_low = z_1, z_top = z_2] 

trgradte.create_data_set("data", df_grad_te )
trTe.create_data_set("data", df_Te )
trne.create_data_set("data", df_ne)
trZ.create_data_set("data", df_Z)
trqe.create_data_set("data", df_qe)
trx.create_data_set("data", df_x)

Data_h5.close()
