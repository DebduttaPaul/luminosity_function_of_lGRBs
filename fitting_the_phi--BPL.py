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
marker_size	=   7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.



z_binned__Fermi	=	np.array( [0, 1.538, 2.657, 10] )
z_binned__Swift	=	np.array( [0, 1.809, 3.455, 10] )


####################################################################################################################################################






####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data


L_vs_z__known_long 	=	ascii.read( './../tables/L_vs_z__known_long.txt' , format = 'fixed_width' )
L_vs_z__Fermi_long 	=	ascii.read( './../tables/L_vs_z__Fermi_long.txt' , format = 'fixed_width' )
L_vs_z__Swift_long 	=	ascii.read( './../tables/L_vs_z__Swift_long.txt' , format = 'fixed_width' )
L_vs_z__other_long 	=	ascii.read( './../tables/L_vs_z__other_long.txt' , format = 'fixed_width' )

known_long_redshift			=	L_vs_z__known_long[ 'measured z'].data
known_long_Luminosity		=	L_vs_z__known_long[ 'Luminosity'].data			* L_norm
known_long_Luminosity_error	=	L_vs_z__known_long[ 'Luminosity error'].data	* L_norm

Fermi_long_redshift			=	L_vs_z__Fermi_long[  'pseudo z' ].data
Fermi_long_Luminosity		=	L_vs_z__Fermi_long[ 'Luminosity'].data
Fermi_long_Luminosity_error	=	L_vs_z__Fermi_long[ 'Luminosity_error'].data

Swift_long_redshift			=	L_vs_z__Swift_long[  'pseudo z' ].data
Swift_long_Luminosity		=	L_vs_z__Swift_long[ 'Luminosity'].data
Swift_long_Luminosity_error	=	L_vs_z__Swift_long[ 'Luminosity_error'].data

other_long_redshift			=	L_vs_z__other_long[ 'measured z'].data
other_long_Luminosity		=	L_vs_z__other_long[ 'Luminosity'].data
other_long_Luminosity_error	=	L_vs_z__other_long[ 'Luminosity_error'].data




inds_to_delete	=	np.where(other_long_Luminosity < 1e-16 )[0]
other_long_redshift			=	np.delete( other_long_redshift  ,       inds_to_delete )
other_long_Luminosity		=	np.delete( other_long_Luminosity,       inds_to_delete )
other_long_Luminosity_error	=	np.delete( other_long_Luminosity_error, inds_to_delete )

inds_to_delete	=	[]
for j, z in enumerate( Swift_long_redshift ):
	array	=	np.abs( z_sim - z )
	ind		=	np.where( array == array.min() )[0]
	if ( Swift_long_Luminosity[j] - L_cut__Swift[ind] ) < 0 :
		inds_to_delete.append( j )
inds_to_delete	=	np.array( inds_to_delete )
Swift_long_redshift			=	np.delete( Swift_long_redshift        , inds_to_delete )
Swift_long_Luminosity		=	np.delete( Swift_long_Luminosity      , inds_to_delete )
Swift_long_Luminosity_error	=	np.delete( Swift_long_Luminosity_error, inds_to_delete )


inds_to_delete				=	np.where(  Fermi_long_redshift > 10 )[0]
Fermi_long_redshift			=	np.delete( Fermi_long_redshift        , inds_to_delete )
Fermi_long_Luminosity		=	np.delete( Fermi_long_Luminosity      , inds_to_delete )
Fermi_long_Luminosity_error	=	np.delete( Fermi_long_Luminosity_error, inds_to_delete )
inds_to_delete				=	np.where(  Swift_long_redshift > 10 )[0]
Swift_long_redshift			=	np.delete( Swift_long_redshift        , inds_to_delete )
Swift_long_Luminosity		=	np.delete( Swift_long_Luminosity      , inds_to_delete )
Swift_long_Luminosity_error	=	np.delete( Swift_long_Luminosity_error, inds_to_delete )
#	print np.where(  known_long_redshift > 10 )[0].size
#	print np.where(  other_long_redshift > 10 )[0].size


#~ sorted_Fermi_redshifts	=	np.sort( Fermi_long_redshift )
#~ equal_bin	=	sorted_Fermi_redshifts.size / 3
#~ print 'Fermi equal bins at', sorted_Fermi_redshifts[int(1*equal_bin)], sorted_Fermi_redshifts[int(2*equal_bin)], sorted_Fermi_redshifts[int(3*equal_bin-1)]
#~ sorted_Swift_redshifts	=	np.sort( Swift_long_redshift )
#~ equal_bin	=	sorted_Swift_redshifts.size / 3
#~ print 'Swift equal bins at', sorted_Swift_redshifts[int(1*equal_bin)], sorted_Swift_redshifts[int(2*equal_bin)], sorted_Swift_redshifts[int(3*equal_bin-1)]
#~ print '\n\n'




x__known_long, y__known_long, y__known_long_poserr, y__known_long_negerr		=	sf.my_histogram_with_errorbars( np.log10(known_long_Luminosity/L_norm), np.log10( (known_long_Luminosity + known_long_Luminosity_error) / L_norm ) - np.log10(known_long_Luminosity/L_norm), np.log10( (known_long_Luminosity + known_long_Luminosity_error) / L_norm ) - np.log10(known_long_Luminosity/L_norm), logL_bin, logL_min, logL_max )
x__Fermi_long, y__Fermi_long, y__Fermi_long_poserr, y__Fermi_long_negerr		=	sf.my_histogram_with_errorbars( np.log10(Fermi_long_Luminosity/L_norm), np.log10( (Fermi_long_Luminosity + Fermi_long_Luminosity_error) / L_norm ) - np.log10(Fermi_long_Luminosity/L_norm), np.log10( (Fermi_long_Luminosity + Fermi_long_Luminosity_error) / L_norm ) - np.log10(Fermi_long_Luminosity/L_norm), logL_bin, logL_min, logL_max )
x__Swift_long, y__Swift_long, y__Swift_long_poserr, y__Swift_long_negerr		=	sf.my_histogram_with_errorbars( np.log10(Swift_long_Luminosity/L_norm), np.log10( (Swift_long_Luminosity + Swift_long_Luminosity_error) / L_norm ) - np.log10(Swift_long_Luminosity/L_norm), np.log10( (Swift_long_Luminosity + Swift_long_Luminosity_error) / L_norm ) - np.log10(Swift_long_Luminosity/L_norm), logL_bin, logL_min, logL_max )
x__other_long, y__other_long, y__other_long_poserr, y__other_long_negerr		=	sf.my_histogram_with_errorbars( np.log10(other_long_Luminosity/L_norm), np.log10( (other_long_Luminosity + other_long_Luminosity_error) / L_norm ) - np.log10(other_long_Luminosity/L_norm), np.log10( (other_long_Luminosity + other_long_Luminosity_error) / L_norm ) - np.log10(other_long_Luminosity/L_norm), logL_bin, logL_min, logL_max )




rho_vs_z__table	=	ascii.read( './../tables/rho_star_dot.txt', format = 'fixed_width' )
z_sim			=	rho_vs_z__table[  'z'].data
rho_star_dot	=	rho_vs_z__table['rho'].data
volume_term		=	rho_vs_z__table['vol'].data


Luminosity_mids		=	x__known_long
Luminosity_mins		=	L_norm	*	(  10 ** ( Luminosity_mids - logL_bin/2 )  )
Luminosity_maxs		=	L_norm	*	(  10 ** ( Luminosity_mids + logL_bin/2 )  )

L_lo	=	Luminosity_mins.min()
L_hi	=	Luminosity_maxs.max()


####################################################################################################################################################





####################################################################################################################################################


def model_evolvingBPL( L_cut, coeff, delta, chi, nu1, nu2, z1, z2 ):
	
	
	CSFR				=	rho_star_dot.copy()
	inds_to_zero		=	np.where( (z_sim<z1) | (z2<z_sim) )[0]
	CSFR[inds_to_zero]	=	0
	CSFR				=	CSFR  *  (  (1+z_sim)**chi  )
	CSFR				=	CSFR  *  volume_term
	
	
	L_b					=	( L_norm * coeff ) * ( (1+z_sim)**delta )
	denominator			=	(  ( 1 - (L_lo/L_b)**(-nu1+1) ) / (-nu1+1)  )  +  (  ( (L_hi/L_b)**(-nu2+1) - 1 ) / (-nu2+1)  )
	
	
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut <= L1 )[0]
		Lmin		=	L_cut.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		ind_low			=	np.where( L_b <= L1 )[0]
		ind_mid			=	np.where( (L1 < L_b) & (L_b < L2) )[0]
		ind_high		=	np.where( L2 <= L_b )[0]
		
		integral_over_L[ind_low]	=	(  ((Lmax/L_b)[ind_low])**(-nu2+1) - ((Lmin/L_b)[ind_low])**(-nu2+1)  ) / (-nu2+1)
		integral_over_L[ind_mid]	=	(  ( 1 - ((Lmin/L_b)[ind_mid])**(-nu1+1) ) / (-nu1+1)  )  +  (  ( ((Lmax/L_b)[ind_mid])**(-nu2+1) - 1 ) / (-nu2+1)  )
		integral_over_L[ind_high]	=	(  ((Lmax/L_b)[ind_high])**(-nu1+1) - ((Lmin/L_b)[ind_high])**(-nu1+1)  ) / (-nu1+1)
		integral_overL				=	integral_over_L / denominator
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		
		integrand	=	CSFR  *  integral_over_L		
		integral	=	simps( integrand, z_sim )
		
		N_vs_L__model[j]	=	integral
	
	
	return N_vs_L__model

def find_discrepancy( model, observed ):
	return np.sum(  ( model - observed ) ** 2  )


####################################################################################################################################################






####################################################################################################################################################


obsFermi_long		=	np.zeros( (z_binned__Fermi.size-1 , Luminosity_mids.size) )
obsFermi_long_poserr=	obsFermi_long.copy()
obsFermi_long_negerr=	obsFermi_long.copy()
obsFermi_long_error	=	obsFermi_long.copy()
obsSwift_long		=	np.zeros( (z_binned__Swift.size-1 , Luminosity_mids.size) )
obsSwift_long_poserr=	obsSwift_long.copy()
obsSwift_long_negerr=	obsSwift_long.copy()
obsSwift_long_error	=	obsSwift_long.copy()
known_long_L	=	known_long_Luminosity.copy()	;	known_long_L_error	=	known_long_Luminosity_error.copy()
Fermi_long_L	=	Fermi_long_Luminosity.copy()	;	Fermi_long_L_error	=	Fermi_long_Luminosity_error.copy()
Swift_long_L	=	known_long_Luminosity.copy()	;	Swift_long_L_error	=	Swift_long_Luminosity_error.copy()
other_long_L	=	other_long_Luminosity.copy()	;	other_long_L_error	=	other_long_Luminosity_error.copy()
for j, z in enumerate( z_binned__Fermi[:-1] ):
	
	inds_to_take	=	np.where(  ( z < known_long_redshift ) & ( known_long_redshift < z_binned__Fermi[j+1] )  )[0]
	known_long_L	=	known_long_Luminosity[inds_to_take]	;	known_long_L_error	=	known_long_Luminosity_error[inds_to_take]
	inds_to_take	=	np.where(  ( z < Fermi_long_redshift ) & ( Fermi_long_redshift < z_binned__Fermi[j+1] )  )[0]
	Fermi_long_L	=	Fermi_long_Luminosity[inds_to_take]	;	Fermi_long_L_error	=	Fermi_long_Luminosity_error[inds_to_take]
	x__known_long_bin, y__known_long_bin, y__known_long_poserr_bin, y__known_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(known_long_L/L_norm), np.log10( (known_long_L + known_long_L_error) / L_norm ) - np.log10(known_long_L/L_norm), np.log10( (known_long_L + known_long_L) / L_norm ) - np.log10(known_long_L/L_norm), logL_bin, logL_min, logL_max )
	x__Fermi_long_bin, y__Fermi_long_bin, y__Fermi_long_poserr_bin, y__Fermi_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(Fermi_long_L/L_norm), np.log10( (Fermi_long_L + Fermi_long_L_error) / L_norm ) - np.log10(Fermi_long_L/L_norm), np.log10( (Fermi_long_L + Fermi_long_L) / L_norm ) - np.log10(Fermi_long_L/L_norm), logL_bin, logL_min, logL_max )
	obsFermi_long_bin		=	y__known_long_bin + y__Fermi_long_bin
	obsFermi_long_poserr_bin=	np.sqrt(  y__known_long_poserr_bin**2 + y__Fermi_long_poserr_bin**2  )
	obsFermi_long_negerr_bin=	np.sqrt(  y__known_long_negerr_bin**2 + y__Fermi_long_negerr_bin**2  )
	obsFermi_long_error_bin	=	np.maximum(obsFermi_long_negerr_bin, obsFermi_long_poserr_bin)+1
	obsFermi_long[j]		=	obsFermi_long_bin
	obsFermi_long_poserr[j]	=	obsFermi_long_poserr_bin
	obsFermi_long_negerr[j]	=	obsFermi_long_negerr_bin
	obsFermi_long_error[j]	=	obsFermi_long_error_bin

for j, z in enumerate( z_binned__Swift[:-1] ):
	inds_to_take	=	np.where(  ( z < Swift_long_redshift ) & ( Swift_long_redshift < z_binned__Swift[j+1] )  )[0]
	Swift_long_L	=	Swift_long_Luminosity[inds_to_take]	;	Swift_long_L_error	=	Swift_long_Luminosity_error[inds_to_take]
	inds_to_take	=	np.where(  ( z < other_long_redshift ) & ( other_long_redshift < z_binned__Swift[j+1] )  )[0]
	other_long_L	=	other_long_Luminosity[inds_to_take]	;	other_long_L_error	=	other_long_Luminosity_error[inds_to_take]
	x__Swift_long_bin, y__Swift_long_bin, y__Swift_long_poserr_bin, y__Swift_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(Swift_long_L/L_norm), np.log10( (Swift_long_L + Swift_long_L_error) / L_norm ) - np.log10(Swift_long_L/L_norm), np.log10( (Swift_long_L + Swift_long_L) / L_norm ) - np.log10(Swift_long_L/L_norm), logL_bin, logL_min, logL_max )
	x__other_long_bin, y__other_long_bin, y__other_long_poserr_bin, y__other_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(other_long_L/L_norm), np.log10( (other_long_L + other_long_L_error) / L_norm ) - np.log10(other_long_L/L_norm), np.log10( (other_long_L + other_long_L) / L_norm ) - np.log10(other_long_L/L_norm), logL_bin, logL_min, logL_max )
	obsSwift_long_bin		=	y__Swift_long_bin + y__other_long_bin
	obsSwift_long_poserr_bin=	np.sqrt(  y__Swift_long_poserr_bin**2 + y__other_long_poserr_bin**2  )
	obsSwift_long_negerr_bin=	np.sqrt(  y__Swift_long_negerr_bin**2 + y__other_long_negerr_bin**2  )
	obsSwift_long_error_bin	=	np.maximum(obsSwift_long_negerr_bin, obsSwift_long_poserr_bin)+1
	obsSwift_long[j]		=	obsSwift_long_bin
	obsSwift_long_poserr[j]	=	obsSwift_long_poserr_bin
	obsSwift_long_negerr[j]	=	obsSwift_long_negerr_bin
	obsSwift_long_error[j]	=	obsSwift_long_error_bin





#	nu1_min	=	0.50	;	nu1_max	=	1.00	;	nu1_bin	=	0.10	#3
#	nu2_min	=	2.00	;	nu2_max	=	2.50	;	nu2_bin	=	0.10	#3
#	Lb__min	=	0.05	;	Lb__max	=	1.80	;	Lb__bin	=	0.25	#3
#	del_min	=	0.85	;	del_max	=	2.60	;	del_bin	=	0.25	#3
#	chi_min	=	-0.40	;	chi_max	=	-0.15	;	chi_bin	=	0.05	#3
#	nu1_min	=	0.40	;	nu1_max	=	0.90	;	nu1_bin	=	0.10	#4
#	nu2_min	=	2.00	;	nu2_max	=	2.50	;	nu2_bin	=	0.10	#4
#	Lb__min	=	0.05	;	Lb__max	=	2.80	;	Lb__bin	=	0.25	#4
#	del_min	=	0.10	;	del_max	=	3.10	;	del_bin	=	0.25	#4
#	chi_min	=	-0.40	;	chi_max	=	-0.15	;	chi_bin	=	0.05	#4

print '...Fermi...', '\n\n'
#~ nu1_min	=	0.10	;	nu1_max	=	1.60	;	nu1_bin	=	0.25	#1
#~ nu2_min	=	1.50	;	nu2_max	=	2.60	;	nu2_bin	=	0.25	#1
#~ Lb__min	=	0.05	;	Lb__max	=	1.50	;	Lb__bin	=	0.25	#1
#~ del_min	=	1.10	;	del_max	=	4.50	;	del_bin	=	0.25	#1
#~ chi_min	=	-2.00	;	chi_max	=	-0.05	;	chi_bin	=	0.25	#1
#~ nu1_min	=	0.40	;	nu1_max	=	0.90	;	nu1_bin	=	0.10	#2
#~ nu2_min	=	2.30	;	nu2_max	=	2.80	;	nu2_bin	=	0.10	#2
#~ Lb__min	=	0.10	;	Lb__max	=	0.60	;	Lb__bin	=	0.10	#2
#~ del_min	=	2.60	;	del_max	=	3.11	;	del_bin	=	0.10	#2
#~ chi_min	=	-1.00	;	chi_max	=	-0.40	;	chi_bin	=	0.10	#2
#~ nu1_min	=	0.50	;	nu1_max	=	0.75	;	nu1_bin	=	0.05	#3
#~ nu2_min	=	2.50	;	nu2_max	=	3.00	;	nu2_bin	=	0.10	#3
#~ Lb__min	=	0.20	;	Lb__max	=	0.45	;	Lb__bin	=	0.05	#3
#~ del_min	=	2.80	;	del_max	=	3.00	;	del_bin	=	0.05	#3
#~ chi_min	=	-1.00	;	chi_max	=	-0.75	;	chi_bin	=	0.05	#3
#~ nu1_min	=	0.55	;	nu1_max	=	0.80	;	nu1_bin	=	0.05	#4
#~ nu2_min	=	2.30	;	nu2_max	=	3.10	;	nu2_bin	=	0.10	#4
#~ Lb__min	=	0.20	;	Lb__max	=	0.45	;	Lb__bin	=	0.05	#4
#~ del_min	=	2.80	;	del_max	=	3.00	;	del_bin	=	0.05	#4
#~ chi_min	=	-0.90	;	chi_max	=	-0.70	;	chi_bin	=	0.05	#4
#~ nu1_min	=	0.60	;	nu1_max	=	0.701	;	nu1_bin	=	0.05	#5
#~ nu2_min	=	2.30	;	nu2_max	=	3.85	;	nu2_bin	=	0.25	#5
#~ Lb__min	=	0.25	;	Lb__max	=	0.351	;	Lb__bin	=	0.05	#5
#~ del_min	=	2.85	;	del_max	=	2.951	;	del_bin	=	0.05	#5
#~ chi_min	=	-0.85	;	chi_max	=	-0.749	;	chi_bin	=	0.05	#5
#~ nu1_min	=	0.60	;	nu1_max	=	0.701	;	nu1_bin	=	0.05	#6
#~ nu2_min	=	3.00	;	nu2_max	=	3.25	;	nu2_bin	=	0.05	#6
#~ Lb__min	=	0.25	;	Lb__max	=	0.351	;	Lb__bin	=	0.05	#6
#~ del_min	=	2.85	;	del_max	=	2.951	;	del_bin	=	0.05	#6
#~ chi_min	=	-0.85	;	chi_max	=	-0.749	;	chi_bin	=	0.05	#6
#~ nu1_min	=	0.15	;	nu1_max	=	1.151	;	nu1_bin	=	0.25	#7, parameter-error estimation
#~ nu2_min	=	2.60	;	nu2_max	=	3.601	;	nu2_bin	=	0.25	#7, parameter-error estimation
#~ Lb__min	=	0.05	;	Lb__max	=	0.801	;	Lb__bin	=	0.25	#7, parameter-error estimation
#~ del_min	=	2.40	;	del_max	=	3.401	;	del_bin	=	0.25	#7, parameter-error estimation
#~ chi_min	=	-1.30	;	chi_max	=	-0.29	;	chi_bin	=	0.25	#7, parameter-error estimation
#~ nu1_min	=	0.25	;	nu1_max	=	0.851	;	nu1_bin	=	0.10	#8, parameter-error estimation
#~ nu2_min	=	2.60	;	nu2_max	=	3.101	;	nu2_bin	=	0.125	#8, parameter-error estimation
#~ Lb__min	=	0.20	;	Lb__max	=	0.50	;	Lb__bin	=	0.05	#8, parameter-error estimation
#~ del_min	=	2.90	;	del_max	=	2.901	;	del_bin	=	0.25	#8, parameter-error estimation
#~ chi_min	=	-1.80	;	chi_max	=	-0.04	;	chi_bin	=	0.25	#8, parameter-error estimation
nu1_min	=	0.65	;	nu1_max	=	0.651	;	nu1_bin	=	0.05	#9, parameter-error estimation
nu2_min	=	3.10	;	nu2_max	=	4.101	;	nu2_bin	=	0.25	#9, parameter-error estimation
Lb__min	=	0.30	;	Lb__max	=	0.301	;	Lb__bin	=	0.05	#9, parameter-error estimation
del_min	=	2.90	;	del_max	=	2.901	;	del_bin	=	0.05	#9, parameter-error estimation
chi_min	=	-2.00	;	chi_max	=	-1.80	;	chi_bin	=	0.05	#9, parameter-error estimation
nu1_array	=	np.arange( nu1_min, nu1_max, nu1_bin )	;	nu1_size	=	nu1_array.size
nu2_array	=	np.arange( nu2_min, nu2_max, nu2_bin )	;	nu2_size	=	nu2_array.size
Lb__array	=	np.arange( Lb__min, Lb__max, Lb__bin )	;	Lb__size	=	Lb__array.size
del_array	=	np.arange( del_min, del_max, del_bin )	;	del_size	=	del_array.size
chi_array	=	np.arange( chi_min, chi_max, chi_bin )	;	chi_size	=	chi_array.size
print 'nu1_array:	', nu1_array
print 'nu2_array:	', nu2_array
print 'Lb__array:	', Lb__array
print 'del_array:	', del_array
print 'chi_array:	', chi_array, '\n'
grid_of_discrepancy__Fermi	=	np.zeros( (nu1_size, nu2_size, Lb__size, del_size, chi_size) )
grid_of_rdcdchisqrd__Fermi	=	grid_of_discrepancy__Fermi.copy()
print 'Grid of {0:d} (nu1) X {1:d} (nu2) X {2:d} (Lb) X {3:d} (del) X {4:d} (chi) = {5:d}.'.format(nu1_size, nu2_size, Lb__size, del_size, chi_size, grid_of_rdcdchisqrd__Fermi.size)






t0	=	time.time()
for c1, nu1 in enumerate(nu1_array):
	for c2, nu2 in enumerate(nu2_array):
		for cLb, coeff in enumerate(Lb__array):
			for cdel, delta in enumerate(del_array):
				for cX, chi in enumerate(chi_array):
					
					modeled__Fermi_long			=	np.zeros( (z_binned__Fermi.size-1 , Luminosity_mids.size) )
					discrepancy_over_z__Fermi	=	np.zeros( z_binned__Fermi.size )
					rdcdchisqrd_over_z__Fermi	=	discrepancy_over_z__Fermi.copy()
					
					#	ax	=	{}
					#	fig	=	plt.figure( figsize=(6, 6) )
					
					for j, z in enumerate( z_binned__Fermi[:-1] ):
						
						if j == 0:
							model_bin__Fermi	=	model_evolvingBPL( L_cut__Fermi, coeff, delta, chi, nu1, nu2, z, z_binned__Fermi[j+1] )
							Fermi_norm			=	obsFermi_long[j].sum() / model_bin__Fermi.sum()
						
						model_bin__Fermi	=	model_evolvingBPL( L_cut__Fermi, coeff, delta, chi, nu1, nu2, z, z_binned__Fermi[j+1] )
						model_bin__Fermi	=	model_bin__Fermi * Fermi_norm
						
						#	ax[j] = fig.add_subplot(z_binned__Fermi.size-1,1,j+1)
						#	ax[j].text( -3.5, 350, r'$ z : $ ' + r'$ {0:.1f} $'.format(z) + r' $ \rm{to} $ ' + r'$ {0:.1f} $'.format(z_binned__Fermi[j+1]), fontsize = size_font, ha = 'center', va = 'center' )
						#	ax[j].set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
						#	ax[j].set_ylim( 0, 400 )
						#	ax[j].errorbar( Luminosity_mids, obsFermi_long[j], yerr = [ obsFermi_long_poserr[j], obsFermi_long_negerr[j] ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
						#	ax[j].plot( Luminosity_mids, model_bin__Fermi, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
						#	ax[j].legend()
						#	ltext = plt.gca().get_legend().get_texts()
						#	plt.setp( ltext[0], fontsize = 11 )
						
						modeled__Fermi_long[j]			=	model_bin__Fermi
						discrepancy_over_z__Fermi[j]	=	find_discrepancy( model_bin__Fermi, obsFermi_long[j] )
						rdcdchisqrd_over_z__Fermi[j]	=	mf.reduced_chisquared( model_bin__Fermi, obsFermi_long[j], obsFermi_long_error[j], 5 )[2]
					
						#	print mf.reduced_chisquared( model_bin__Fermi, obsFermi_long[j], obsFermi_long_error[j], 5 )[1]
					
					#	xticklabels	=	ax[0].get_xticklabels() + ax[1].get_xticklabels()
					#	plt.setp( xticklabels, visible = False )
					#	plt.xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font, labelpad = padding+5 )
					#	plt.show()
					
					grid_of_discrepancy__Fermi[c1, c2, cLb, cdel, cX]	=	np.sum(  discrepancy_over_z__Fermi )
					grid_of_rdcdchisqrd__Fermi[c1, c2, cLb, cdel, cX]	=	np.mean( rdcdchisqrd_over_z__Fermi )
print 'Fermi done in {:.3f} hrs.'.format( ( time.time()-t0 )/3600 ), '\n'

#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--1.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--2.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--3.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--4.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--5.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--6.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--7.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--rdcdchisqrd--8.pkl', 'wb' )
output = open( './../tables/pkl/Fermi--rdcdchisqrd--9.pkl', 'wb' )
pickle.dump( grid_of_rdcdchisqrd__Fermi, output )
output.close()

#~ output = open( './../tables/pkl/Fermi--discrepancy--1.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--2.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--3.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--4.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--5.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--6.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--7.pkl', 'wb' )
#~ output = open( './../tables/pkl/Fermi--discrepancy--8.pkl', 'wb' )
output = open( './../tables/pkl/Fermi--discrepancy--9.pkl', 'wb' )
pickle.dump( grid_of_discrepancy__Fermi, output )
output.close()


ind_discrepancy_min__Fermi	=	np.unravel_index( grid_of_discrepancy__Fermi.argmin(), grid_of_discrepancy__Fermi.shape )
nu1__Fermi	=	nu1_array[ind_discrepancy_min__Fermi[0]]
nu2__Fermi	=	nu2_array[ind_discrepancy_min__Fermi[1]]
Lb___Fermi	=	Lb__array[ind_discrepancy_min__Fermi[2]]
del__Fermi	=	del_array[ind_discrepancy_min__Fermi[3]]
chi__Fermi	=	chi_array[ind_discrepancy_min__Fermi[4]]
print 'Minimum discrepancy of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}, delta = {4:.2f}, chi = {5:.2f}'.format( grid_of_discrepancy__Fermi[ind_discrepancy_min__Fermi], nu1__Fermi, nu2__Fermi, Lb___Fermi, del__Fermi, chi__Fermi )
print 'Reduced-chisquared of {0:.3f}.'.format( grid_of_rdcdchisqrd__Fermi[ind_discrepancy_min__Fermi]), '\n'

ind_rdcdchisqrd_min__Fermi	=	np.unravel_index( grid_of_rdcdchisqrd__Fermi.argmin(), grid_of_rdcdchisqrd__Fermi.shape )
nu1__Fermi	=	nu1_array[ind_rdcdchisqrd_min__Fermi[0]]
nu2__Fermi	=	nu2_array[ind_rdcdchisqrd_min__Fermi[1]]
Lb___Fermi	=	Lb__array[ind_rdcdchisqrd_min__Fermi[2]]
del__Fermi	=	del_array[ind_rdcdchisqrd_min__Fermi[3]]
chi__Fermi	=	chi_array[ind_rdcdchisqrd_min__Fermi[4]]
print 'Minimum reduced-chisquared of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}, delta = {4:.2f}, chi = {5:.2f}'.format( grid_of_rdcdchisqrd__Fermi[ind_rdcdchisqrd_min__Fermi], nu1__Fermi, nu2__Fermi, Lb___Fermi, del__Fermi, chi__Fermi ), '\n\n\n\n\n\n\n\n'








#~ #	nu1_min	=	0.60	;	nu1_max	=	1.09	;	nu1_bin	=	0.10	#3
#~ #	nu2_min	=	2.00	;	nu2_max	=	2.50	;	nu2_bin	=	0.10	#3
#~ #	Lb__min	=	0.01	;	Lb__max	=	0.30	;	Lb__bin	=	0.05	#3
#~ #	del_min	=	3.05	;	del_max	=	4.80	;	del_bin	=	0.25	#3
#~ #	chi_min	=	-1.80	;	chi_max	=	-0.60	;	chi_bin	=	0.25	#3
#~ #	nu1_min	=	0.60	;	nu1_max	=	1.09	;	nu1_bin	=	0.10	#4
#~ #	nu2_min	=	2.00	;	nu2_max	=	2.50	;	nu2_bin	=	0.10	#4
#~ #	Lb__min	=	0.01	;	Lb__max	=	0.45	;	Lb__bin	=	0.05	#4
#~ #	del_min	=	2.55	;	del_max	=	5.05	;	del_bin	=	0.25	#4
#~ #	chi_min	=	-1.80	;	chi_max	=	-0.60	;	chi_bin	=	0.25	#4
#~ 
print '...Swift...', '\n\n'
nu1_min	=	0.10	;	nu1_max	=	1.60	;	nu1_bin	=	0.25	#1
nu2_min	=	1.50	;	nu2_max	=	2.60	;	nu2_bin	=	0.25	#1
Lb__min	=	0.05	;	Lb__max	=	1.50	;	Lb__bin	=	0.25	#1
del_min	=	1.10	;	del_max	=	4.50	;	del_bin	=	0.25	#1
chi_min	=	-2.00	;	chi_max	=	-0.05	;	chi_bin	=	0.25	#1
nu1_min	=	0.40	;	nu1_max	=	0.90	;	nu1_bin	=	0.10	#2
nu2_min	=	1.50	;	nu2_max	=	2.01	;	nu2_bin	=	0.10	#2
Lb__min	=	0.01	;	Lb__max	=	0.26	;	Lb__bin	=	0.05	#2
del_min	=	3.90	;	del_max	=	4.39	;	del_bin	=	0.10	#2
chi_min	=	-1.20	;	chi_max	=	-0.70	;	chi_bin	=	0.10	#2
nu1_min	=	0.50	;	nu1_max	=	1.00	;	nu1_bin	=	0.10	#3
nu2_min	=	1.80	;	nu2_max	=	2.30	;	nu2_bin	=	0.10	#3
Lb__min	=	0.03	;	Lb__max	=	0.10	;	Lb__bin	=	0.01	#3
del_min	=	3.90	;	del_max	=	4.20	;	del_bin	=	0.10	#3
chi_min	=	-1.00	;	chi_max	=	-0.50	;	chi_bin	=	0.10	#3
nu1_min	=	0.50	;	nu1_max	=	1.00	;	nu1_bin	=	0.10	#4
nu2_min	=	1.80	;	nu2_max	=	2.50	;	nu2_bin	=	0.10	#4
Lb__min	=	0.05	;	Lb__max	=	0.10	;	Lb__bin	=	0.01	#4
del_min	=	3.90	;	del_max	=	4.20	;	del_bin	=	0.05	#4
chi_min	=	-0.90	;	chi_max	=	-0.50	;	chi_bin	=	0.10	#4
nu1_min	=	0.60	;	nu1_max	=	0.801	;	nu1_bin	=	0.10	#5
nu2_min	=	1.90	;	nu2_max	=	2.40	;	nu2_bin	=	0.10	#5
Lb__min	=	0.06	;	Lb__max	=	0.09	;	Lb__bin	=	0.01	#5
del_min	=	3.90	;	del_max	=	4.05	;	del_bin	=	0.05	#5
chi_min	=	-0.80	;	chi_max	=	-0.30	;	chi_bin	=	0.10	#5
nu1_min	=	0.60	;	nu1_max	=	0.801	;	nu1_bin	=	0.05	#6
nu2_min	=	2.00	;	nu2_max	=	2.25	;	nu2_bin	=	0.05	#6
Lb__min	=	0.06	;	Lb__max	=	0.09	;	Lb__bin	=	0.01	#6
del_min	=	3.90	;	del_max	=	4.05	;	del_bin	=	0.05	#6
chi_min	=	-0.70	;	chi_max	=	-0.45	;	chi_bin	=	0.05	#6
nu1_min	=	0.40	;	nu1_max	=	0.90	;	nu1_bin	=	0.10	#7
nu2_min	=	1.50	;	nu2_max	=	3.80	;	nu2_bin	=	0.25	#7
Lb__min	=	0.10	;	Lb__max	=	0.60	;	Lb__bin	=	0.10	#7
del_min	=	2.70	;	del_max	=	3.30	;	del_bin	=	0.10	#7
chi_min	=	-1.00	;	chi_max	=	-0.40	;	chi_bin	=	0.10	#7
nu1_min	=	0.60	;	nu1_max	=	0.81	;	nu1_bin	=	0.05	#8
nu2_min	=	2.50	;	nu2_max	=	3.01	;	nu2_bin	=	0.10	#8
Lb__min	=	0.10	;	Lb__max	=	0.35	;	Lb__bin	=	0.05	#8
del_min	=	3.00	;	del_max	=	3.25	;	del_bin	=	0.05	#8
chi_min	=	-0.60	;	chi_max	=	-0.10	;	chi_bin	=	0.10	#8
nu1_array	=	np.arange( nu1_min, nu1_max, nu1_bin )	;	nu1_size	=	nu1_array.size
nu2_array	=	np.arange( nu2_min, nu2_max, nu2_bin )	;	nu2_size	=	nu2_array.size
Lb__array	=	np.arange( Lb__min, Lb__max, Lb__bin )	;	Lb__size	=	Lb__array.size
del_array	=	np.arange( del_min, del_max, del_bin )	;	del_size	=	del_array.size
chi_array	=	np.arange( chi_min, chi_max, chi_bin )	;	chi_size	=	chi_array.size
print 'nu1_array:	', nu1_array
print 'nu2_array:	', nu2_array
print 'Lb__array:	', Lb__array
print 'del_array:	', del_array
print 'chi_array:	', chi_array, '\n'
grid_of_discrepancy__Swift	=	np.zeros( (nu1_size, nu2_size, Lb__size, del_size, chi_size) )
grid_of_rdcdchisqrd__Swift	=	grid_of_discrepancy__Swift.copy()
print 'Grid of {0:d} (nu1) X {1:d} (nu2) X {2:d} (Lb) X {3:d} (del) X {4:d} (chi) = {5:d}.'.format(nu1_size, nu2_size, Lb__size, del_size, chi_size, grid_of_rdcdchisqrd__Swift.size)

t0	=	time.time()
for c1, nu1 in enumerate(nu1_array):
	for c2, nu2 in enumerate(nu2_array):
		for cLb, coeff in enumerate(Lb__array):
			for cdel, delta in enumerate(del_array):
				for cX, chi in enumerate(chi_array):
					
					modeled__Swift_long			=	np.zeros( (z_binned__Swift.size-1 , Luminosity_mids.size) )
					discrepancy_over_z__Swift	=	np.zeros( z_binned__Swift.size )
					rdcdchisqrd_over_z__Swift	=	discrepancy_over_z__Swift.copy()
					
					#	ax	=	{}
					#	fig	=	plt.figure( figsize=(6, 6) )
					
					for j, z in enumerate( z_binned__Swift[:-1] ):
						
						if j == 0:
							model_bin__Swift	=	model_evolvingBPL( L_cut__Swift, coeff, delta, chi, nu1, nu2, z, z_binned__Swift[j+1] )
							Swift_norm			=	obsSwift_long[j].sum() / model_bin__Swift.sum()
						
						model_bin__Swift	=	model_evolvingBPL( L_cut__Swift, coeff, delta, chi, nu1, nu2, z, z_binned__Swift[j+1] )
						model_bin__Swift	=	model_bin__Swift * Swift_norm
						
						#	ax[j] = fig.add_subplot(z_binned__Swift.size-1,1,j+1)
						#	ax[j].text( -3.5, 350, r'$ z : $ ' + r'$ {0:.1f} $'.format(z) + r' $ \rm{to} $ ' + r'$ {0:.1f} $'.format(z_binned__Swift[j+1]), fontsize = size_font, ha = 'center', va = 'center' )
						#	ax[j].set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
						#	ax[j].set_ylim( 0, 400 )
						#	ax[j].errorbar( Luminosity_mids, obsSwift_long[j], yerr = [ obsSwift_long_poserr[j], obsSwift_long_negerr[j] ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
						#	ax[j].plot( Luminosity_mids, model_bin__Swift, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
						#	ax[j].legend()
						#	ltext = plt.gca().get_legend().get_texts()
						#	plt.setp( ltext[0], fontsize = 11 )
						
						modeled__Swift_long[j]			=	model_bin__Swift
						discrepancy_over_z__Swift[j]	=	find_discrepancy( model_bin__Swift, obsSwift_long[j] )
						rdcdchisqrd_over_z__Swift[j]	=	mf.reduced_chisquared( model_bin__Swift, obsSwift_long[j], obsSwift_long_error[j], 5 )[2]
					
					#	xticklabels	=	ax[0].get_xticklabels() + ax[1].get_xticklabels()
					#	plt.setp( xticklabels, visible = False )
					#	plt.xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font, labelpad = padding+5 )
					#	plt.show()
					
					grid_of_discrepancy__Swift[c1, c2, cLb, cdel, cX]	=	np.sum(  discrepancy_over_z__Swift )
					grid_of_rdcdchisqrd__Swift[c1, c2, cLb, cdel, cX]	=	np.mean( rdcdchisqrd_over_z__Swift )
print 'Swift done in {:.3f} hrs.'.format( ( time.time()-t0 )/3600 ), '\n'
#~ 
output = open( './../tables/pkl/Swift--rdcdchisqrd--1.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--2.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--3.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--4.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--5.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--6.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--7.pkl', 'wb' )
output = open( './../tables/pkl/Swift--rdcdchisqrd--8.pkl', 'wb' )
pickle.dump( grid_of_rdcdchisqrd__Swift, output )
output.close()
#~ 
output = open( './../tables/pkl/Swift--discrepancy--1.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--2.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--3.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--4.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--5.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--6.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--7.pkl', 'wb' )
output = open( './../tables/pkl/Swift--discrepancy--8.pkl', 'wb' )
pickle.dump( grid_of_discrepancy__Swift, output )
output.close()


ind_discrepancy_min__Swift	=	np.unravel_index( grid_of_discrepancy__Swift.argmin(), grid_of_discrepancy__Swift.shape )
nu1__Swift	=	nu1_array[ind_discrepancy_min__Swift[0]]
nu2__Swift	=	nu2_array[ind_discrepancy_min__Swift[1]]
Lb___Swift	=	Lb__array[ind_discrepancy_min__Swift[2]]
del__Swift	=	del_array[ind_discrepancy_min__Swift[3]]
chi__Swift	=	chi_array[ind_discrepancy_min__Swift[4]]
print 'Minimum discrepancy of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}, delta = {4:.2f}, chi = {5:.2f}'.format( grid_of_discrepancy__Swift[ind_discrepancy_min__Swift], nu1__Swift, nu2__Swift, Lb___Swift, del__Swift, chi__Swift )
print 'Reduced-chisquared of {0:.3f}.'.format( grid_of_rdcdchisqrd__Swift[ind_discrepancy_min__Swift] ), '\n'

ind_rdcdchisqrd_min__Swift	=	np.unravel_index( grid_of_rdcdchisqrd__Swift.argmin(), grid_of_rdcdchisqrd__Swift.shape )
nu1__Swift	=	nu1_array[ind_rdcdchisqrd_min__Swift[0]]
nu2__Swift	=	nu2_array[ind_rdcdchisqrd_min__Swift[1]]
Lb___Swift	=	Lb__array[ind_rdcdchisqrd_min__Swift[2]]
del__Swift	=	del_array[ind_rdcdchisqrd_min__Swift[3]]
chi__Swift	=	chi_array[ind_rdcdchisqrd_min__Swift[4]]
print 'Minimum reduced-chisquared of {0:.3f} at nu1 = {1:.2f}, nu2 = {2:.2f}, Lb = {3:.2f}, delta = {4:.2f}, chi = {5:.2f}'.format( grid_of_rdcdchisqrd__Swift[ind_rdcdchisqrd_min__Swift], nu1__Swift, nu2__Swift, Lb___Swift, del__Swift, chi__Swift )


