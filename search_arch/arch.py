## all same blocks
all_SCB_k3 = ['SCB_k3']*10
all_SCB_k5 = ['SCB_k5']*10
all_SCB_k7 = ['SCB_k7']*10
all_SRB_k3 = ['SRB_k3']*10
all_SRB_k5 = ['SRB_k5']*10
all_SRB_k7 = ['SRB_k7']*10
all_SIB_k3_e1 = ['SIB_k3_e1']*10
all_SIB_k3_e3 = ['SIB_k3_e3']*10
all_SIB_k5_e1 = ['SIB_k5_e1']*10
all_SIB_k5_e3 = ['SIB_k5_e3']*10

## Our architecture 
AutoSNN = ['SRB_k5', 'max_pool_k2', 'SRB_k5', 'skip_connect', 'max_pool_k2', 'SRB_k3', 'SRB_k5', 'max_pool_k2']

## AutoSNN with different lambda (Table 2)
AutoSNN_0 = ['SRB_k5', 'max_pool_k2', 'SRB_k5', 'SRB_k5', 'max_pool_k2', 'SCB_k5', 'SRB_k5', 'max_pool_k2'] # lambda = 0
AutoSNN_16 = ['SRB_k5', 'max_pool_k2', 'SRB_k5', 'skip_connect', 'max_pool_k2', 'SRB_k5', 'SRB_k5', 'max_pool_k2'] # lambda = -0.16
AutoSNN_24 = ['skip_connect', 'max_pool_k2', 'SRB_k5', 'skip_connect', 'max_pool_k2', 'SCB_k5', 'SRB_k5', 'max_pool_k2'] # lambda = -0.24

## 10 randomly sampled architecture (Random sampling in Table 5)
random_1 = ['SCB_k3','SRB_k5','SRB_k3','skip_connect','SRB_k5','skip_connect','SRB_k5','SRB_k5']
random_2 = ['SRB_k5','SRB_k5','skip_connect','SCB_k5','SRB_k5','SCB_k5','skip_connect','SRB_k5']
random_3 = ['SCB_k3','SRB_k5','SCB_k5','SRB_k3','SRB_k5','SCB_k5','SCB_k5','SRB_k5']
random_4 = ['SCB_k5','SRB_k5','SCB_k3','skip_connect','SRB_k5','SCB_k3','SCB_k5','SRB_k5']
random_5 = ['SRB_k5','SRB_k5','SCB_k3','skip_connect','SRB_k5','skip_connect','SRB_k3','SRB_k5']
random_6 = ['SCB_k3','SRB_k5','SRB_k5','SRB_k3','SRB_k5','SCB_k3','SRB_k3','SRB_k5']
random_7 = ['SCB_k5','SRB_k5','SCB_k3','SRB_k5','SRB_k5','SCB_k3','SCB_k5','SRB_k5']
random_8 = ['SCB_k5','SRB_k5','skip_connect','skip_connect','SRB_k5','SCB_k5','SRB_k3','SRB_k5']
random_9 = ['SCB_k3','SRB_k5','SRB_k3','SRB_k3','SRB_k5','SCB_k5','SRB_k3','SRB_k5']
random_10 = ['skip_connect','SRB_k5','SRB_k3','skip_connect','SRB_k5','SRB_k3','SCB_k5','SRB_k5']

## WS + random search (in Table 5)
random_search_0 = ['SCB_k5', 'max_pool_k2', 'SCB_k5', 'SRB_k5', 'max_pool_k2', 'SCB_k5', 'SRB_k5', 'max_pool_k2'] # lambda = 0
random_search_8 = ['SRB_k5', 'max_pool_k2', 'SRB_k5', 'SRB_k3', 'max_pool_k2', 'SCB_k3', 'SRB_k5', 'max_pool_k2'] # lambda = -0.08

## Architecture search without spiking neurons (Table 7)
AutoSNN_ANN_space = ['SRB_k5', 'max_pool_k2', 'SRB_k3', 'SCB_k3', 'max_pool_k2', 'SCB_k5', 'SRB_k5', 'max_pool_k2']

