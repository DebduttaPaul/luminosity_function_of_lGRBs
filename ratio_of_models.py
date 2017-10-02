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
import time
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
erg_per_keV	=	1.6022 * 1e-9

logL_bin	=	0.5
logL_min	=	-5
logL_max	=	+5

GBM_sensitivity	=	1e-8 * 8.0	# in erg.s^{-1}.cm^{2}.
BAT_sensitivity	=	0.20		# in  ph.s^{-1}.cm^{2}.


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	2e+1 #	for the purposes of plotting
x_in_keV_min	=	1e00	;	x_in_keV_max	=	2e04	#	Ep(1+z), min & max.
y_in_eps_min	=	1e49	;	y_in_eps_max	=	1e56	#	L_iso  , min & max.


####################################################################################################################################################



####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data


k_data	=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim	=	k_data['z'].data
k_Fermi	=	k_data['k_Fermi'].data
k_Swift	=	k_data['k_Swift'].data



L_vs_z__known_long 	=	ascii.read( './../tables/L_vs_z__known_long.txt' , format = 'fixed_width' )

known_long_redshift			=	L_vs_z__known_long[ 'measured z'].data
known_long_Luminosity		=	L_vs_z__known_long[ 'Luminosity'].data			* L_norm
known_long_Luminosity_error	=	L_vs_z__known_long[ 'Luminosity error'].data	* L_norm

x__known_long, y__known_long, y__known_long_poserr, y__known_long_negerr		=	sf.my_histogram_with_errorbars( np.log10(known_long_Luminosity/L_norm), np.log10( (known_long_Luminosity + known_long_Luminosity_error) / L_norm ) - np.log10(known_long_Luminosity/L_norm), np.log10( (known_long_Luminosity + known_long_Luminosity_error) / L_norm ) - np.log10(known_long_Luminosity/L_norm), logL_bin, logL_min, logL_max )

Luminosity_mids		=	x__known_long
Luminosity_mins		=	L_norm  *  (  10 ** ( Luminosity_mids - logL_bin/2 )  )
Luminosity_maxs		=	L_norm  *  (  10 ** ( Luminosity_mids + logL_bin/2 )  )
Luminosity_mids		=	L_norm  *  (  10 ** Luminosity_mids )
L_lo	=	Luminosity_mins.min()
L_hi	=	Luminosity_maxs.max()


####################################################################################################################################################



####################################################################################################################################################


def lumfunc( coeff, delta, chi, nu1, nu2 ):
	
	luminosity_function	=	np.zeros( (Luminosity_mids.size, z_sim.size) )
	L_b	=	np.ones(z_sim.size) * ( L_norm * coeff ) * ( (1+z_sim)**delta )
	denominator	=	(  ( 1 - (L_lo/L_b)**(-nu1+1) ) / (-nu1+1)  )  +  (  ( (L_hi/L_b)**(-nu2+1) - 1 ) / (-nu2+1)  )


	for j, L in enumerate( Luminosity_mids ):
		for k, z in enumerate( z_sim ):
			
			L_break	=	L_b[k]
			
			if L <= L_break:	luminosity_function[j,k]	=	(L/L_break)**(-nu1)
			else:				luminosity_function[j,k]	=	(L/L_break)**(-nu2)
		
		luminosity_function[j,:] = luminosity_function[j,:] / denominator
	
	
	return luminosity_function


####################################################################################################################################################






####################################################################################################################################################


#~ ##	2017-05-31, #4
#~ nu1__Fermi		=	0.65
#~ nu2__Fermi		=	3.10
#~ coeff__Fermi	=	0.30
#~ delta__Fermi	=	2.90
#~ chi__Fermi		=	-0.80
#~ 
#~ nu1__Swift		=	0.70
#~ nu2__Swift		=	2.10
#~ coeff__Swift	=	0.07
#~ delta__Swift	=	3.95
#~ chi__Swift		=	-0.60



##	2017-05-31, #6
nu1__Fermi		=	0.65
nu2__Fermi		=	3.10
coeff__Fermi	=	0.30
delta__Fermi	=	2.90
chi__Fermi		=	-0.80


nu1__Swift		=	0.70
nu2__Swift		=	2.10
coeff__Swift	=	0.07
delta__Swift	=	3.95
chi__Swift		=	-0.65



####################################################################################################################################################




####################################################################################################################################################



epsilon			=	10.0
factor			=	3.0



ratio_of_kS		=	( (k_Swift*erg_per_keV*BAT_sensitivity) / (k_Fermi*GBM_sensitivity) ) ** epsilon


lumfunc__Fermi	=	lumfunc( coeff__Fermi, delta__Fermi, chi__Fermi, nu1__Fermi, nu2__Fermi )
lumfunc__Swift	=	lumfunc( coeff__Swift, delta__Swift, chi__Swift, nu1__Swift, nu2__Swift )
ratio_of_phiS	=	lumfunc__Swift / lumfunc__Fermi


L_prime		=	L_norm / factor
ind			=	mf.nearest( L_cut__Fermi   , L_prime )
m			=	mf.nearest( Luminosity_mids, L_prime )
z_sim__cut	=	z_sim[0:ind]

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.plot( z_sim__cut, ratio_of_phiS[m, 0:ind], label = r'$\phi$' )
ax.plot( z_sim__cut, ratio_of_kS  [   0:ind], label = r'$ k $' )
ax.legend()
plt.savefig( './../plots/estimated_lumfunc_models/ratio.png' )
plt.clf()
plt.close()
