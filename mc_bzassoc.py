from itertools import chain, combinations
import numpy as np
import pandas as pd
import networkx as nx
from math import log

class Country:
	def __init__(self, name, matrix_index, weight=0, population=0, area=0):
		self.name = name
		self.matrix_index = matrix_index
		self.weight = weight
		self.population = population
		self.area = area		
		self.bz_critical = 0.0  # number of times the country is critical
		self.bz_assoc_critical = 0.0 # number of times the country is critical in association model
		self.bz_index = 0.0
		self.bz_index_assoc = 0.0
	
	def display(self):
		print self.name+", "+"weight: "+str(self.weight) + " population: "+str(self.population)+" area: "+ str(self.area)

def powerset(iterable):
    '''
    	eg: powerset of the list [1,2,3]
    	powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)    
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def random_association_matrix(n):
	'''
	 return nxn matrix with diagonal entries as 1
	 and rest as values b/w (-1,1)
	'''
	a = 2 * np.random.rand(n, n) - 1
	np.fill_diagonal(a, 1)
	return a

def winning_coaltions(country_list, quota):
	'''		
		input: 
		country_list: list of country objects
		weight_quota : the net weight required for a coalition to be winning
		population_quota : the net population required for a coalition to be winning

		returns: a list of all the winning coalitions.
	'''
	winning_coaltions=[]
	for coalition in powerset(country_list):
		coalition_stat = np.sum([[c.weight, c.population, c.area] for c in coalition], axis=0)
		if (coalition_stat >= quota).all():
			winning_coaltions.append(coalition)
	return winning_coaltions

def get_banzaf(all_countries, winning_coaltions, quota, assoc_matrix):
	'''
		input:
		all_countries: list of all the country objects
		winning_coaltiont: list of all the winning coaltions
		assoc_matrix: a numpy ndarray with entries(b/w -1,1) as association between countries

		returms bz normalized_without association and bz_with association
	'''	
	all_countries_numeric = np.array([[c.weight, c.population, c.area] for c in all_countries])
	for coalition in winning_coaltions:
		coalition_stat = np.sum([[c.weight,c.population,c.area] for c in coalition], axis=0)		
		for country in coalition:
			if (coalition_stat - [country.weight, country.population, country.area] < quota).any():
				country.bz_critical += 1
			if (coalition_stat - np.dot(assoc_matrix[country.matrix_index], all_countries_numeric) < quota).any():
				country.bz_assoc_critical += 1
	
	bz = np.array([[c.bz_critical, c.bz_assoc_critical] for c in all_countries])
	return bz / 2**(len(all_countries)-1) # absolute banzhaf indices, number of subsets in which the player i is critical. all subsets containing i 2^(n-1)

def read_country_data(input_directory, input_file_name):
	'''
		reads country data and returns a list of country objects
		input_directory: str, and it has xlsx extension
		input_file_name: str
	'''
	df = pd.read_excel(input_directory + input_file_name)
	country_list = [Country(df.iloc[i,:].country, i, df.iloc[i,:].weight, df.iloc[i,:].population, df.iloc[i,:].area) for i in range(0,df.shape[0])]
	return country_list

def read_association_matrix(input_directory, input_file_name):
	'''
		if we wish to set a non random association matrix
		input_file_name: str, and it has xlsx extension
	'''
	df = pd.read_excel(input_directory + input_file_name, header=None)
	return np.array(df)

def get_nx_graph(country_list, adjacency_matrix):
	'''
		i/p
		country_list
		adjacency_matrix: numpy nd array, here we use association matrix as adjacency matrix

		returns:networkx directed graph with nodes as countries and edges as the corresponding values in association matrix
	'''
	dg = nx.DiGraph()
	dg.add_nodes_from(country_list)
	for country1 in country_list:
		for country2 in country_list:
			full_precision = adjacency_matrix[country1.matrix_index][country2.matrix_index]
			dg.add_edge(country1, country2, weight=float("{0:.2f}".format(full_precision))) # put only 2 decimal precision
	return dg

def draw_nx_graph(nx_graph, output_file):
    """ 
        Generates graphs (png image) from networkx graph object

    Args:
        nx_graph(str): networkx graph object to be drawn
        output_drectory(str): location to save graph
        output_file_name(str): name of the image file to be saved

    Returns:
       null

    """
    mapping = {}
    for u in nx_graph.nodes():
    	mapping[u] = u.name +"  "+str(u.bz_index)+ "  " + str(u.bz_index_assoc) 

    for u, v, d in nx_graph.edges(data=True):        
        d['label'] = d.get('weight', '')

    nx_graph_to_draw = nx.relabel_nodes(nx_graph, mapping)
    A = nx.nx.drawing.nx_agraph.to_agraph(nx_graph_to_draw)
    A.layout(prog='dot')
    A.draw(output_file)

def monte_carlo(cl, quota, assoc_matrix, i, delta, width):
    eps = width / 2
    k = 0
    X = 0.0
    X_assoc = 0.0
    loop_max = log(2/delta) / (2*eps**2)
    country_i_numeric = [cl[i].weight, cl[i].population, cl[i].area]
    all_countries_numeric = np.array([[c.weight, c.population, c.area] for c in cl])
    
    while(k < loop_max):
        players_bits = np.random.randint(2, size = len(cl))
        players_bits[i] = 1
        coalition = np.extract(players_bits, cl)
        col_sum = np.sum([[c.weight,c.population,c.area] for c in coalition], axis=0)
        if (col_sum >= quota).all() and (col_sum - country_i_numeric < quota).any():
            X += 1
        if (col_sum >= quota).all() and (col_sum - np.dot(assoc_matrix[i], all_countries_numeric) < quota).any():
        	X_assoc += 1
        k += 1
        #print k 
    return X / k, X_assoc / k

#cl = read_country_data('data/','fakeEU.xlsx')
#cl = read_country_data('data/','EU_data_altered.xlsx') #exponential time complexity
cl = read_country_data('data/','EU_data.xlsx')
matrix = random_association_matrix(len(cl))
#matrix = read_association_matrix('data/',"fakeasso.xlsx")
## for non_random_association matrix use ##
# matrix = read_association_matrix('data/','assoc_matrix.xlsx')
#quota = [25, 27, 29]
quota = [51, 54, 58]
# wc = winning_coaltions(cl, quota) 
# bz = get_banzaf(cl, wc, quota, matrix)	
# print bz

delta = 0.1
confidence_interval_width = 0.01
for i in range(len(cl)):
    print monte_carlo(cl, quota, matrix, i, delta, confidence_interval_width)
# print matrix

# dg = get_nx_graph(cl, matrix)
# draw_nx_graph(dg,"precsion_graph.svg")
