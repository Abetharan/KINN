import numpy as np 
import h5py
import os 
import matplotlib.pyplot as plt 
import random as rand 
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

def epperlein_short(nx, L, Z_ = 37.25, ne_ = 1e27, Te_ = 100., sin= False):

    x_l = 0
    x_u = L
    initial_coord = np.linspace(x_l, x_u, nx+1, dtype=np.float64)
    centered_coord = np.array([(initial_coord[i] + initial_coord[i+1]) /2 for i in range(len(initial_coord) -1)], dtype=np.float64)
    Z = np.zeros(nx) + Z_#37.25
    
    ne = np.zeros(nx) + ne_# 1e27
    #temperatureE = np.linspace(3, 300, nx)
    
    ##Relevant for periodic systems .. fluid cannot be
    if sin: 
        temperatureE = np.zeros(nx, dtype=np.float64) + Te_  + 1E-3 * Te_  * np.sin((2*np.pi * centered_coord) / np.max(centered_coord), dtype=np.float64)
    else:
    ##Reflective
        temperatureE = np.zeros(nx, dtype=np.float64) + Te_  + 1e-3 * Te_  * np.cos((np.pi * centered_coord) / np.max(centered_coord), dtype=np.float64)

    return(centered_coord, temperatureE, ne, Z)

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



def Ramp(nx, L, lower_cut_off, upper_cut_off, Te_low, Te_up, ne_low, ne_top, z_low, z_top):
    lower_cutof = lower_cut_off
    upper_cutof = upper_cut_off
    x_up = np.linspace(0, lower_cutof*L, int(nx*0.333))
    step = x_up[1] - x_up[0]
    x_ramp = np.linspace(lower_cutof*L + step/10, upper_cutof*L, int(nx*0.33333) + 1)
    step = x_ramp[1] - x_ramp[0]
    x_down = np.linspace(upper_cutof*L+step/10, L, int(0.3333*nx))
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

    Te_Up = Te_up 
    Te_Down = Te_low 
    lower_limit = lower_cutof*L#x_centered[int(nx*0.4)]
    upper_limit = upper_cutof*L#x_centered[int(nx*0.7)]

    Te = smooth(x_centered, Te_Up, Te_Down, lower_limit, upper_limit)
    
    ne_Up = ne_top * 1e26
    ne_Down = ne_low * 1e26
    ne = smooth(x_centered, ne_Up, ne_Down, lower_limit, upper_limit) 

    Z_Up = z_top#2
    Z_Down = z_low#36.64
    Z = smooth(x_centered, Z_Up, Z_Down, lower_limit, upper_limit) 
    
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

def getTwoRandom():
    k = rand.random()
    k_1 = rand.random()
    if k < k_1:
        getTwoRandom() 
    else:
        return(k, k_1)

base_path = "/Users/shiki/DATA/KINN_RELATED"
te_h5 = h5py.File(os.path.join(base_path, "Te.hdf5"), "w")
ne_h5 = h5py.File(os.path.join(base_path, "ne.hdf5"), "w")
z_h5 = h5py.File(os.path.join(base_path, "Z.hdf5"), "w")
x_h5 = h5py.File(os.path.join(base_path, "x.hdf5"), "w")

ne_range = np.linspace(1e19, 1e28, 100) #1e21-1e28
te_range =  np.linspace(1, 3000, 100)# 1ev - 3kev
z_range =   np.linspace(2, 38, 100)#2 - 38
L_range =  np.linspace(3E-3, 3, 100)#3mm - 3m
nx = 101
n = 100
for i in range(30000):
    L = L_range[np.random.randint(0, n)]
    z_1 = z_range[np.random.randint(0, n)]
    z_2 = z_range[np.random.randint(0, n)]
    n_1 = ne_range[np.random.randint(0, n)]
    n_2 = ne_range[np.random.randint(0, n)]
    t_1 = te_range[np.random.randint(0, n)]
    t_2 = te_range[np.random.randint(0, n)]
    lower, upper = getTwoRandom()  
    Te_random = np.random.uniform(0.94, 1.06)
    ne_random = np.random.uniform(0.90, 1.1)
    
    
    x, Te, ne, Z = epperlein_short(nx, L ,z_1, n_1, t_1) 
    x, Te, ne, Z = TwoTemperatureBath(nx, L, z_1, t_1, t_2) 

    x, Te, ne, Z = Ramp(nx, L, lower_cut_off = lower , upper_cut_off = upper, Te_low = t_1, Te_up = t_2, ne_low = t_1, ne_top = t_2 , z_low = z_1, z_top = z_2) 
    x, Te, ne, Z = AllSmoothRamp(nx, L, lower_cutof = lower , upper_cutof = upper, Te_begin = t_2 * upper, Te_end = t_1 * lower, Te_low = t_1, Te_top = t_2,
                                 ne_begin =n_2 * upper , ne_end =n_1*lower, ne_low = n_1, ne_top =n_2 , Z_begin = z_2*upper, Z_end = z_1*lower ,z_low = z_1, z_top = z_2) 
te_h5.close()
ne_h5.close()
z_h5.close()
x_h5.close()