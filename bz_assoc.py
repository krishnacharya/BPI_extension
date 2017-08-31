from itertools import chain, combinations
import numpy as np
import pandas as pd
import networkx as nx

class Country:
	def __init__(self, name, matrix_index, weight=0, population=0,area=0):
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
		coalition_stat = np.sum([[c.weight,c.population,c.area] for c in coalition], axis=0)
		if np.array_equal(coalition_stat >= quota, [True,True,True]):
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
	all_countries_numeric = np.array([[c.weight,c.population,c.area] for c in all_countries])
	for coalition in winning_coaltions:
		coalition_stat = np.sum([[c.weight,c.population,c.area] for c in coalition], axis=0)		
		for country in coalition:
			if not np.array_equal(coalition_stat - [country.weight,country.population,country.area] >= quota, [True,True,True]):
				country.bz_critical += 1
			if not np.array_equal((coalition_stat - np.dot(assoc_matrix[country.matrix_index], all_countries_numeric)) >= quota, [True,True,True]):
				country.bz_assoc_critical += 1
	
	bza = np.array([[c.bz_critical, c.bz_assoc_critical] for c in all_countries])
	return bza / np.sum(bza, axis=0)

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
			dg.add_edge(country1, country2, weight=adjacency_matrix[country1.matrix_index][country2.matrix_index])
	return dg

cl = read_country_data('data/','EU_data.xlsx')
matrix = random_association_matrix(len(cl))

## for non_random_association matrix use ##
# matrix = read_association_matrix('data/','assoc_matrix.xlsx')

quota=[51, 40, 60]
wc=winning_coaltions(cl, quota) 
bza = get_banzaf(cl, wc, quota, matrix)	
print bza