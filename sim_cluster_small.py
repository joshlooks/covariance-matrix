import sys
import numpy as np
import os
from scipy.interpolate import interp1d

def gillespie_meta(N_cities,N_population,init_I,betat,gamma,mu,T_max):
    np.random.seed()
    # Initialize state variables
    I = np.copy(init_I)  # Initial infections per city
    S = np.full(N_cities, N_population) - I  # Initial susceptibles per city

    # Results arrays
    times = np.zeros(70000)
    St = np.zeros((70000,3))
    It = np.zeros((70000,3))
    Ct = np.zeros((70000,3))
    St[0,:] = np.copy(S)
    It[0,:] = np.copy(I)
    C = np.copy(I)
    Ct[0,:] = np.copy(C)

    # Gillespie algorithm
    t = 0
    ind = 0
    while t < T_max:          
        beta_matrix = betat(t)
        # Compute event rates
        infection_rates = (beta_matrix @ I) * S / N_population
        recovery_rates = gamma * I
        birth_death_rates = mu * (N_population-S)
        
        total_rate = np.sum(infection_rates) + np.sum(recovery_rates) + np.sum(birth_death_rates)
        if total_rate == 0:
            break
        
        # Draw time step
        t += np.random.exponential(1 / total_rate)
        ind += 1
        St[ind,:] = St[ind-1,:]
        It[ind,:] = It[ind-1,:]
        # Choose event type (infection, recovery, birth-death pair)
        rand = np.random.rand() * total_rate
        if rand < infection_rates[0]:
            S[0] -= 1
            I[0] += 1
        elif rand < infection_rates[0] + infection_rates[1]:
            S[1] -= 1
            I[1] += 1
        elif rand < np.sum(infection_rates):
            S[2] -= 1
            I[2] += 1
        elif rand < np.sum(infection_rates) + recovery_rates[0]:
            I[0] -= 1
            C[0] += 1
        elif rand < np.sum(infection_rates) + recovery_rates[0] + recovery_rates[1]:
            I[1] -= 1
            C[1] += 1
        elif rand < np.sum(infection_rates) + np.sum(recovery_rates):
            I[2] -= 1
            C[2] += 1
        elif rand < np.sum(infection_rates) + np.sum(recovery_rates) + birth_death_rates[0]:
            if np.random.rand() < I[0]/(N_population - S[0]):
                I[0] -= 1
            S[0] += 1
        elif rand < np.sum(infection_rates) + np.sum(recovery_rates) + birth_death_rates[0] + birth_death_rates[1]:
            if np.random.rand() < I[1]/(N_population - S[1]):
                I[1] -= 1
            S[1] += 1
        else:
            if np.random.rand() < I[2]/(N_population - S[2]):
                I[2] -= 1
            S[2] += 1
        # Store state
        St[ind,:] = S
        It[ind,:] = I
        Ct[ind,:] = C
        times[ind] = t
    return times[:ind+1], St[:ind+1], It[:ind+1], Ct[:ind+1]

identity = sys.argv[2]
num = int(sys.argv[1])
script_dir = os.path.dirname(__file__)

# Parameters
N_cities = 3   # Number of cities
T_max = 100    # Maximum time
b_w = 0.231  # Transmission within each city
b_b = 0.12 # Transmission between cities
beta_matrix = np.array([[b_w,b_b,0],[b_b,b_w,b_b],[0,b_b,b_w]])
gamma = 0.1    # Recovery rate
mu = 1/(80*365)     # Birth and death rate
N_population = 10000  # Population per city
Ii = np.array([100,100,100])
Si = N_population - Ii

# Transmission functions
betat = lambda t: beta_matrix
def beta_indec_t(t):
    if t < 5:
        return beta_matrix
    else:
        temp = np.diag([0,1,0])
        return beta_matrix - 0.1*(t-5)*temp if (beta_matrix - 0.1*(t-5)*temp >= 0).all() else beta_matrix - np.diag([0,beta_matrix[1,1],0])
    
def beta_betdec_t(t):
    if t < 15:
        return beta_matrix
    else:
        temp = np.array([[0,1,0],[1,0,1],[0,1,0]])
        return beta_matrix - 0.1*(t-15)*temp if (beta_matrix - 0.1*(t-15)*temp >= 0).all() else beta_matrix - np.array([[0,b_b,0],[b_b,0,b_b],[0,b_b,0]])
    
def beta_newvar_t(t):
    b_w_l = 0.05  # Transmission within each city
    b_b_l = 0.025 # Transmission between cities
    beta_matrix_low = np.array([[b_w_l,b_b_l,0],[b_b_l,b_w_l,b_b_l],[0,b_b_l,b_w_l]])
    if t < 15:
        return beta_matrix_low
    else:
        temp = np.array([[0,0,0],[0,0,0.25],[0,0,1]])
        return beta_matrix_low + 0.005*(t-15)*temp

# Setup sims and output files
if identity == 'constant':
    results_dir = os.path.join(script_dir,'constant_small')
    beta_t = betat
    def sim():
        return gillespie_meta(N_cities,N_population,Ii,beta_t,gamma,mu,T_max)
elif identity == 'indec':
    results_dir = os.path.join(script_dir,'indec_small')
    beta_t = beta_indec_t
    def sim():
        return gillespie_meta(N_cities,N_population,Ii,beta_t,gamma,mu,T_max)
elif identity == 'betdec':
    results_dir = os.path.join(script_dir,'betdec_small')
    beta_t = beta_betdec_t
    def sim():
        return gillespie_meta(N_cities,N_population,Ii,beta_t,gamma,mu,T_max)
elif identity == 'newvar':
    results_dir = os.path.join(script_dir,'newvar_small')
    beta_t = beta_newvar_t
    def sim():
        return gillespie_meta(N_cities,N_population,Ii,beta_t,gamma,mu,T_max)
else:
    results_dir = os.path.join(script_dir,'seed_small')
    beta_t = betat
    Ii = np.array([0,0,100])
    Si = N_population - Ii
    def sim():
        return gillespie_meta(N_cities,N_population,Ii,beta_t,gamma,mu,T_max)

# Run number of sims
times = np.arange(0,100.1,0.1)
num_sims = 100
Is = np.zeros((num_sims,3,len(times)))
Ss = np.zeros((num_sims,3,len(times)))
Cs = np.zeros((num_sims,3,len(times)))
for i in range(100):
    res = sim()
    Is[i,:,:] = np.array([np.interp(times,res[0],res[2][:,i]) for i in range(3)])
    Ss[i,:,:] = np.array([np.interp(times,res[0],res[1][:,i]) for i in range(3)])
    Cs[i,:,:] = np.array([np.interp(times,res[0],res[3][:,i]) for i in range(3)])

i = np.random.randint(100)

fpath_I = os.path.join(results_dir,f'I_{num}_{i}.npy')
fpath_S = os.path.join(results_dir,f'S_{num}_{i}.npy')
fpath_C = os.path.join(results_dir,f'C_{num}_{i}.npy')

np.save(fpath_I,Is)
np.save(fpath_S,Ss)
np.save(fpath_C,Cs)