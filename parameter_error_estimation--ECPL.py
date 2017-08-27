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
marker_size	=	07	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



z_binned__Fermi	=	np.array( [0, 1.538, 2.657, 10] )
z_binned__Swift	=	np.array( [0, 1.809, 3.455, 10] )



#~ ##	2017-08-23, #1
nu__Fermi		=	0.60
Lb__Fermi	=	5.10

nu__Swift		=	0.70
Lb__Swift	=	5.10

####################################################################################################################################################








####################################################################################################################################################


#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--1.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--2.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--3.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--4.pkl', 'rb' )
pkl_file	=	open( './../tables/pkl/Fermi--discrepancy--5.pkl', 'rb' )
Fermi__grid_of_discrepancy	=	pickle.load(pkl_file)
pkl_file.close()
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--1.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--2.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--3.pkl', 'rb' )
#~ pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--4.pkl', 'rb' )
pkl_file	=	open( './../tables/pkl/Fermi--rdcdchisqrd--5.pkl', 'rb' )
Fermi__grid_of_chisquared	=	pickle.load(pkl_file) * 18 * 3
pkl_file.close()


#~ Fermi__nu_min	=	0.10	;	Fermi__nu_max	=	2.10	;	Fermi__nu_bin	=	0.10	#1
#~ Fermi__Lb_min	=	3.20	;	Fermi__Lb_max	=	5.20	;	Fermi__Lb_bin	=	0.10	#1
#~ Fermi__nu_min	=	0.10	;	Fermi__nu_max	=	1.10	;	Fermi__nu_bin	=	0.10	#2
#~ Fermi__Lb_min	=	4.20	;	Fermi__Lb_max	=	6.20	;	Fermi__Lb_bin	=	0.10	#2
#~ Fermi__nu_min	=	0.40	;	Fermi__nu_max	=	0.90	;	Fermi__nu_bin	=	0.10	#3
#~ Fermi__Lb_min	=	5.10	;	Fermi__Lb_max	=	5.70	;	Fermi__Lb_bin	=	0.10	#3
Fermi__nu_min	=	0.60	;	Fermi__nu_max	=	0.70	;	Fermi__nu_bin	=	0.10	#4
Fermi__Lb_min	=	3.40	;	Fermi__Lb_max	=	7.90	;	Fermi__Lb_bin	=	0.50	#4

Fermi__nu_array	=	np.arange( Fermi__nu_min, Fermi__nu_max, Fermi__nu_bin )	;	Fermi__nu_size	=	Fermi__nu_array.size
Fermi__Lb_array	=	np.arange( Fermi__Lb_min, Fermi__Lb_max, Fermi__Lb_bin )	;	Fermi__Lb_size	=	Fermi__Lb_array.size

ind_discrepancy_min__Fermi	=	np.unravel_index( Fermi__grid_of_discrepancy.argmin(), Fermi__grid_of_discrepancy.shape )
nu__Fermi	=	Fermi__nu_array[ind_discrepancy_min__Fermi[0]]
Lb__Fermi	=	Fermi__Lb_array[ind_discrepancy_min__Fermi[1]]

print nu__Fermi, Lb__Fermi
print Fermi__grid_of_discrepancy.shape, ind_discrepancy_min__Fermi

discrepancy_at_solution	=	Fermi__grid_of_discrepancy[ind_discrepancy_min__Fermi]
chisquared_at_solution	=	Fermi__grid_of_chisquared[ind_discrepancy_min__Fermi]
chisquared_for_1sigma	=	chisquared_at_solution + 2.30
Fermi__grid_of_discrepancy	=	Fermi__grid_of_discrepancy * (chisquared_at_solution/discrepancy_at_solution)

print chisquared_for_1sigma
print Fermi__grid_of_discrepancy[ :, ind_discrepancy_min__Fermi[1] ]
print Fermi__grid_of_discrepancy[ ind_discrepancy_min__Fermi[0], : ]
#~ print Fermi__grid_of_chisquared[ :, ind_discrepancy_min__Fermi[1] ]
#~ print Fermi__grid_of_chisquared[ ind_discrepancy_min__Fermi[0], : ]

levels = np.arange( 0, 2, 0.1 )
#~ plt.contourf( Fermi__Lb_array, Fermi__nu_array, Fermi__grid_of_chisquared )
plt.contourf( Fermi__Lb_array, Fermi__nu_array, Fermi__grid_of_discrepancy/18 )
#~ plt.contourf( Fermi__Lb_array, Fermi__nu_array, Fermi__grid_of_discrepancy/18, levels = levels )
plt.plot( Lb__Fermi, nu__Fermi, 'x', color = 'r', mew = 2, ms = 7 )
plt.colorbar()
plt.xlabel( r'$ L_{b,0} $', fontsize = size_font+2, labelpad = padding-2 )
plt.ylabel( r'$ \nu  $', fontsize = size_font+2, labelpad = padding )
plt.title( r'$ Fermi $' )
plt.show()
