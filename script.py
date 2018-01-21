import bzext as pc
import numpy as np

def print_countries(indices):
	for i in range(l):
		print country_list[i].name,bz_exact[i]

country_list = pc.read_country_data('data/Association-Matrix/','EU-wei-pop.xlsx')
matrix = pc.read_association_matrix_shifted('data/Association-Matrix/','assocmatbig.xlsx') #persuation power matrix
#matrix = random_association_matrix(l) #for random matrix
l = len(country_list)

net_sum = np.sum([[c.weight,c.population,c.val] for c in country_list], axis=0)
weight_x = [c.weight for c in country_list]
population_x = [c.population for c in country_list]
quota = [int(net_sum[0] * 0.74), int(net_sum[1] * 0.62), l/2 + 1] # [weight quota, popln quota, majority of the #of countries]


'''
	In the following calculations the normalized index is used as it helps
	compare the values without and with the association matrix
'''

#########EXACT CALCULATION############
# wc = pc.winning_coaltions(country_list, quota)
# bz_exact = pc.get_banzaf(country_list, wc, quota, matrix)	
# print "Exact"
# print_countries(bz_exact)

#########MONTE CARLO ESTIMATE##########
# delta = 0.1
# confidence_interval_width = 0.01
# bz_approx, bz_temp = [], []

# for i in range(l):
#     dat = pc.monte_carlo(country_list, quota, matrix, i, delta, confidence_interval_width)
#     bz_temp.append(dat)
# bz_approx = bz_temp / np.sum(bz_temp, axis=0)
# print "Approx" 
# print_countries(bz_approx)

#######PLOT OF WITHOUT and WITHOUT ASSOCIATION MATRIX#######

# pc.plot_new_vs_old(bz_exact[:,0], bz_exact[:,1], weight_x, xlabel = 'weight')
# pc.plot_new_vs_old(bz_exact[:,0], bz_exact[:,1], population_x, xlabel='population')

