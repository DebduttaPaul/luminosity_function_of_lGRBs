from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.stats import pearsonr as R
from scipy.stats import spearmanr as S
from scipy.stats import kendalltau as T
from scipy.optimize import curve_fit
from scipy.integrate import quad, simps
import debduttaS_functions as mf
import specific_functions as sf
import time, pickle, pprint
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


P	=	np.pi		# Dear old pi!
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
cm_per_Mpc	=	3.0857 * 1e24

logL_bin	=	0.5
logL_min	=	-5
logL_max	=	+5


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



z_binned__Fermi	=	np.array( [0, 1.538, 2.657, 10] )
z_binned__Swift	=	np.array( [0, 1.809, 3.455, 10] )



##	2017-06-01, #6
#~ nu1__Fermi		=	0.65
#~ nu2__Fermi		=	3.10
#~ coeff__Fermi	=	0.30
#~ delta__Fermi	=	2.90
#~ chi__Fermi		=	-0.80
#~ 
#~ nu1__Swift		=	0.65	# Fermi best-fits
#~ nu2__Swift		=	3.10	# Fermi best-fits
#~ coeff__Swift	=	0.30	# Fermi best-fits
#~ delta__Swift	=	2.90	# Fermi best-fits
#~ chi__Swift		=	-0.80	# Fermi best-fits


####################################################################################################################################################






####################################################################################################################################################


def nearest( array, value ):
	
	
	'''
	
	
	Parameters
	-----------
	array:	An array, which is to be searched over.
	value:	The value that is being searched in "array".
	
	Returns
	-----------
	index:	The index in "array" corresponding to which element is the closest to "value".
	
	
	Assumptions
	-----------
	"value" lies close to at least one element in "array".
	
	
	'''
	
	
	#~ diff	=	np.abs( array - value )
	diff	=	( array - value )
	
	print diff
	
	minimum	=	np.min( diff )
	index	=	np.where( diff == minimum )[0]
	
	return index


####################################################################################################################################################







####################################################################################################################################################


#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--1.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--4.pkl', 'rb' )
pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--7.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--8.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--9.pkl', 'rb' )
Fermi__grid_of_discrepancy	=	pickle.load(pkl_file)
pkl_file.close()
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--1.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--4.pkl', 'rb' )
pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--7.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--8.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--9.pkl', 'rb' )
Fermi__grid_of_chisquared	=	pickle.load(pkl_file) * 15 * 3
pkl_file.close()


#~ Fermi__nu1_min	=	0.10	;	Fermi__nu1_max	=	1.60	;	Fermi__nu1_bin	=	0.25	#1
#~ Fermi__nu2_min	=	1.50	;	Fermi__nu2_max	=	2.60	;	Fermi__nu2_bin	=	0.25	#1
#~ Fermi__Lb__min	=	0.05	;	Fermi__Lb__max	=	1.50	;	Fermi__Lb__bin	=	0.25	#1
#~ Fermi__del_min	=	1.10	;	Fermi__del_max	=	4.50	;	Fermi__del_bin	=	0.25	#1
#~ Fermi__chi_min	=	-2.00	;	Fermi__chi_max	=	-0.05	;	Fermi__chi_bin	=	0.25	#1
#~ Fermi__nu1_min	=	0.55	;	Fermi__nu1_max	=	0.80	;	Fermi__nu1_bin	=	0.05	#4
#~ Fermi__nu2_min	=	2.30	;	Fermi__nu2_max	=	3.10	;	Fermi__nu2_bin	=	0.10	#4
#~ Fermi__Lb__min	=	0.20	;	Fermi__Lb__max	=	0.45	;	Fermi__Lb__bin	=	0.05	#4
#~ Fermi__del_min	=	2.80	;	Fermi__del_max	=	3.00	;	Fermi__del_bin	=	0.05	#4
#~ Fermi__chi_min	=	-0.90	;	Fermi__chi_max	=	-0.70	;	Fermi__chi_bin	=	0.05	#4
Fermi__nu1_min	=	0.15	;	Fermi__nu1_max	=	1.151	;	Fermi__nu1_bin	=	0.25	#7
Fermi__nu2_min	=	2.60	;	Fermi__nu2_max	=	3.601	;	Fermi__nu2_bin	=	0.25	#7
Fermi__Lb__min	=	0.05	;	Fermi__Lb__max	=	0.801	;	Fermi__Lb__bin	=	0.25	#7
Fermi__del_min	=	2.40	;	Fermi__del_max	=	3.401	;	Fermi__del_bin	=	0.25	#7
Fermi__chi_min	=	-1.30	;	Fermi__chi_max	=	-0.29	;	Fermi__chi_bin	=	0.25	#7
#~ Fermi__nu1_min	=	0.25	;	Fermi__nu1_max	=	0.851	;	Fermi__nu1_bin	=	0.10	#8
#~ Fermi__nu2_min	=	2.60	;	Fermi__nu2_max	=	3.101	;	Fermi__nu2_bin	=	0.125	#8
#~ Fermi__Lb__min	=	0.20	;	Fermi__Lb__max	=	0.50	;	Fermi__Lb__bin	=	0.05	#8
#~ Fermi__del_min	=	2.90	;	Fermi__del_max	=	2.901	;	Fermi__del_bin	=	0.25	#8
#~ Fermi__chi_min	=	-1.80	;	Fermi__chi_max	=	-0.04	;	Fermi__chi_bin	=	0.25	#8
#~ Fermi__nu1_min	=	0.65	;	Fermi__nu1_max	=	0.651	;	Fermi__nu1_bin	=	0.05	#9
#~ Fermi__nu2_min	=	3.10	;	Fermi__nu2_max	=	4.101	;	Fermi__nu2_bin	=	0.25	#9
#~ Fermi__Lb__min	=	0.30	;	Fermi__Lb__max	=	0.301	;	Fermi__Lb__bin	=	0.05	#9
#~ Fermi__del_min	=	2.90	;	Fermi__del_max	=	2.901	;	Fermi__del_bin	=	0.05	#9
#~ Fermi__chi_min	=	-2.00	;	Fermi__chi_max	=	-1.80	;	Fermi__chi_bin	=	0.05	#9

Fermi__nu1_array	=	np.arange( Fermi__nu1_min, Fermi__nu1_max, Fermi__nu1_bin )	;	Fermi__nu1_size	=	Fermi__nu1_array.size
Fermi__nu2_array	=	np.arange( Fermi__nu2_min, Fermi__nu2_max, Fermi__nu2_bin )	;	Fermi__nu2_size	=	Fermi__nu2_array.size
Fermi__Lb__array	=	np.arange( Fermi__Lb__min, Fermi__Lb__max, Fermi__Lb__bin )	;	Fermi__Lb__size	=	Fermi__Lb__array.size
Fermi__del_array	=	np.arange( Fermi__del_min, Fermi__del_max, Fermi__del_bin )	;	Fermi__del_size	=	Fermi__del_array.size
Fermi__chi_array	=	np.arange( Fermi__chi_min, Fermi__chi_max, Fermi__chi_bin )	;	Fermi__chi_size	=	Fermi__chi_array.size

ind_discrepancy_min__Fermi	=	np.unravel_index( Fermi__grid_of_discrepancy.argmin(), Fermi__grid_of_discrepancy.shape )
nu1__Fermi	=	Fermi__nu1_array[ind_discrepancy_min__Fermi[0]]
nu2__Fermi	=	Fermi__nu2_array[ind_discrepancy_min__Fermi[1]]
Lb___Fermi	=	Fermi__Lb__array[ind_discrepancy_min__Fermi[2]]
del__Fermi	=	Fermi__del_array[ind_discrepancy_min__Fermi[3]]
chi__Fermi	=	Fermi__chi_array[ind_discrepancy_min__Fermi[4]]

print nu1__Fermi, nu2__Fermi, Lb___Fermi, del__Fermi, chi__Fermi
print Fermi__grid_of_discrepancy.shape, ind_discrepancy_min__Fermi

chisquared_at_solution	=	Fermi__grid_of_chisquared[ind_discrepancy_min__Fermi]
chisquared_for_1sigma	=	chisquared_at_solution + 5.89

print chisquared_for_1sigma
#~ print Fermi__grid_of_discrepancy[ :, ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
#~ print Fermi__grid_of_discrepancy[ ind_discrepancy_min__Fermi[0], :, ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
#~ print Fermi__grid_of_discrepancy[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], :, ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
#~ print Fermi__grid_of_discrepancy[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], :, ind_discrepancy_min__Fermi[4] ]
#~ print Fermi__grid_of_discrepancy[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], : ]
print Fermi__grid_of_chisquared[ :, ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
print Fermi__grid_of_chisquared[ ind_discrepancy_min__Fermi[0], :, ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
print Fermi__grid_of_chisquared[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], :, ind_discrepancy_min__Fermi[3], ind_discrepancy_min__Fermi[4] ]
print Fermi__grid_of_chisquared[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], :, ind_discrepancy_min__Fermi[4] ]
print Fermi__grid_of_chisquared[ ind_discrepancy_min__Fermi[0], ind_discrepancy_min__Fermi[1], ind_discrepancy_min__Fermi[2], ind_discrepancy_min__Fermi[3], : ]

