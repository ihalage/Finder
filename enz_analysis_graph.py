import pandas as pd
from pymatgen import Composition
import numpy as np
import itertools

df = pd.read_csv('Promising_ENZ_Finder.csv')


elem_combs = {}
def analyse_comp(row):
	comp = Composition(row.formula)
	elements = [e.name for e in comp.elements]
	combs = itertools.combinations(elements, 2)
	for c in combs:
		if c not in elem_combs.keys():
			elem_combs[c] = 1
		else:
			elem_combs[c] +=1

	return 1




df['dummy'] = df.apply(analyse_comp, axis=1)

ecombs = np.array(list(elem_combs.keys()))
print(ecombs)
frm = ecombs[:,0]
to = ecombs[:,1]
weights = np.array(list(elem_combs.values()))
print(weights)

dfenz = pd.DataFrame({'From': frm, 'To': to, 'Weight': weights})
dfenz = dfenz[dfenz.Weight>=5]
print(dfenz)
dfenz.to_csv('ENZ_analysis_graph5.csv', index=False)
# print(elem_combs)
# print(elem_combs.values())
# print(len(elem_combs.keys()))