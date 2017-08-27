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

logL_bin	=	0.5
logL_min	=	-5
logL_max	=	+5


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	07	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	2e+1 #	for the purposes of plotting



z_binned__Fermi	=	np.array( [0, 1.538, 2.657, 10] )
z_binned__Swift	=	np.array( [0, 1.809, 3.455, 10] )


####################################################################################################################################################






####################################################################################################################################################


def straight_line( x, m, c ):
	return m*x + c
	
def f(x, nu):
	return x**(-nu) * np.exp(-x)


####################################################################################################################################################






####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data
L_cut__CZTI		=	threshold_data['L_cut__CZTI'].data


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


sorted_Fermi_redshifts	=	np.sort( Fermi_long_redshift )
equal_bin	=	sorted_Fermi_redshifts.size / 3
sorted_Swift_redshifts	=	np.sort( Swift_long_redshift )
equal_bin	=	sorted_Swift_redshifts.size / 3
#~ print 'Fermi equal bins at', sorted_Fermi_redshifts[int(1*equal_bin)], sorted_Fermi_redshifts[int(2*equal_bin)], sorted_Fermi_redshifts[int(3*equal_bin-1)]
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


####################################################################################################################################################




####################################################################################################################################################


def plot_Fermi():
	
	print obsFermi_long.sum() / modeled__Fermi_long.sum()
	
	fig	=	plt.figure()
	ax	=	fig.add_subplot(111)
	ax.set_title( r'$ Fermi $' )
	ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font )
	ax.set_ylabel( r'$ \rm{ normalized \; N } $', fontsize = size_font )
	ax.errorbar( Luminosity_mids, obsFermi_long/obsFermi_long.sum(), yerr = [ obsFermi_long_poserr/obsFermi_long.sum(), obsFermi_long_negerr/obsFermi_long.sum() ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
	ax.plot( Luminosity_mids, modeled__Fermi_long/modeled__Fermi_long.sum(), linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
	ax.legend()

def plot_Swift():
	
	print obsSwift_long.sum() / modeled__Swift_long.sum()
	
	fig	=	plt.figure()
	ax	=	fig.add_subplot(111)
	ax.set_title( r'$ Swift $' )
	ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font )
	ax.set_ylabel( r'$ \rm{ normalized \; N  $', fontsize = size_font )
	ax.errorbar( Luminosity_mids, obsSwift_long/obsSwift_long.sum(), yerr = [ obsSwift_long_poserr/obsSwift_long.sum(), obsSwift_long_negerr/obsSwift_long.sum() ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
	ax.plot( Luminosity_mids, modeled__Swift_long/modeled__Swift_long.sum(), linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
	ax.legend()

def find_discrepancy( model, observed ):
	return np.sum(  ( model - observed ) ** 2  )


def model_evolvingexponentialcutoffpowerlaw( L_cut, coeff, delta, nu, chi, z1, z2 ):
	
	
	L_b	=	( L_norm * coeff ) * ( (1+z_sim)**delta )
	N_vs_L__model	=	np.zeros(Luminosity_mids.size)
	
	for j, L1 in enumerate( Luminosity_mins ):
		
		inds		=	np.where( L_cut <= L1 )[0]
		Lmin		=	L_cut.copy()
		Lmin[inds]	=	L1
		
		L2			=	Luminosity_maxs[j]
		Lmax		=	L2 * np.ones(z_sim.size)
		
		integral_over_L	=	L_b.copy()
		for k, z in enumerate(z_sim):
			
			L					=	np.linspace( Lmin[k], Lmax[k], 1e3 )
			integrand			=	(  (L/L_b[k])**(-nu)  )  *  np.exp(-L/L_b[k])
			
			lower_limit			=	L_lo / L_b[k]
			upper_limit			=	L_hi / L_b[k]
			denominator			=	L_b[k] * quad( f, lower_limit, upper_limit, args=(nu) )[0]
			
			integral_over_L[k]	=	simps( integrand, L ) / denominator
		
		
		ind	=	np.where( integral_over_L <= 0  )[0]
		integral_over_L[ind]	=	0
		
		CSFR				=	rho_star_dot.copy()
		inds_to_take		=	np.where( (z1<z_sim) & (z_sim<z2) )[0]
		inds_to_zero		=	np.delete( np.arange(z_sim.size), inds_to_take )
		CSFR[inds_to_zero]	=	0
		CSFR				=	CSFR  *  (  (1+z_sim)**chi  )
		
		integrand	=	CSFR  *  volume_term  *  integral_over_L		
		integral	=	simps( integrand, z_sim )
		
		N_vs_L__model[j]	=	integral
	
	
	return N_vs_L__model



####################################################################################################################################################





####################################################################################################################################################



coeff = 4.1	;	nu	= 0.6	;	chi = 0.0
#~ coeff = 4.1	;	delta = 2.9	;	nu = 0.6	;	chi = -0.8
#~ coeff = 4.1	;	delta = 0.5	;	nu = 0.6	;	chi = 0.0
#~ coeff = 4.1	;	delta = 0.0	;	nu = 0.6	;	chi = -0.8
#~ coeff = 4.1	;	delta = 0.0	;	nu = 0.6	;	chi = 0.0
coeff = 5.40	;	delta = 0.0	;	nu = 0.60	;	chi = 0.0



#~ print 'Fermi...'
#~ known_long_L	=	known_long_Luminosity.copy()	;	known_long_L_error	=	known_long_Luminosity_error.copy()
#~ Fermi_long_L	=	Fermi_long_Luminosity.copy()	;	Fermi_long_L_error	=	Fermi_long_Luminosity_error.copy()
#~ Fermi_long_total_data		=	np.zeros( Luminosity_mids.size )
#~ Fermi_long_total_data_ep	=	np.zeros( Luminosity_mids.size )
#~ Fermi_long_total_data_en	=	np.zeros( Luminosity_mids.size )
#~ Fermi_long_total_model		=	np.zeros( Luminosity_mids.size )
#~ modeled__Fermi_long			=	np.zeros( (z_binned__Fermi.size-1 , Luminosity_mids.size) )
#~ discrepancy_over_z__Fermi	=	np.zeros( z_binned__Fermi.size )
#~ rdcdchisqrd_over_z__Fermi	=	discrepancy_over_z__Fermi.copy()
#~ 
#~ ax	=	{}
#~ fig	=	plt.figure( figsize=(6, 6) )
#~ fig.suptitle( r'$ Fermi $', fontsize = size_font-2 )
#~ 
#~ t0	=	time.time()
#~ for j, z in enumerate( z_binned__Fermi[:-1] ):
	#~ 
	#~ inds_to_take	=	np.where(  ( z < known_long_redshift ) & ( known_long_redshift < z_binned__Fermi[j+1] )  )[0]
	#~ known_long_L	=	known_long_Luminosity[inds_to_take]	;	known_long_L_error	=	known_long_Luminosity_error[inds_to_take]
	#~ inds_to_take	=	np.where(  ( z < Fermi_long_redshift ) & ( Fermi_long_redshift < z_binned__Fermi[j+1] )  )[0]
	#~ Fermi_long_L	=	Fermi_long_Luminosity[inds_to_take]	;	Fermi_long_L_error	=	Fermi_long_Luminosity_error[inds_to_take]
	#~ x__known_long_bin, y__known_long_bin, y__known_long_poserr_bin, y__known_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(known_long_L/L_norm), np.log10( (known_long_L + known_long_L_error) / L_norm ) - np.log10(known_long_L/L_norm), np.log10( (known_long_L + known_long_L) / L_norm ) - np.log10(known_long_L/L_norm), logL_bin, logL_min, logL_max )
	#~ x__Fermi_long_bin, y__Fermi_long_bin, y__Fermi_long_poserr_bin, y__Fermi_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(Fermi_long_L/L_norm), np.log10( (Fermi_long_L + Fermi_long_L_error) / L_norm ) - np.log10(Fermi_long_L/L_norm), np.log10( (Fermi_long_L + Fermi_long_L) / L_norm ) - np.log10(Fermi_long_L/L_norm), logL_bin, logL_min, logL_max )
	#~ obsFermi_long_bin		=	y__known_long_bin + y__Fermi_long_bin
	#~ obsFermi_long_poserr_bin=	np.sqrt(  y__known_long_poserr_bin**2 + y__Fermi_long_poserr_bin**2  )
	#~ obsFermi_long_negerr_bin=	np.sqrt(  y__known_long_negerr_bin**2 + y__Fermi_long_negerr_bin**2  )
	#~ obsFermi_long_error_bin	=	np.maximum(obsFermi_long_negerr_bin, obsFermi_long_poserr_bin)+1
	#~ Fermi_long_total_data	=	Fermi_long_total_data	+ obsFermi_long_bin
	#~ Fermi_long_total_data_ep=	np.sqrt( Fermi_long_total_data_ep**2 + obsFermi_long_poserr_bin**2 )
	#~ Fermi_long_total_data_en=	np.sqrt( Fermi_long_total_data_en**2 + obsFermi_long_negerr_bin**2 )
	#~ 
	#~ if j == 0:
		#~ model_bin__Fermi	=	model_evolvingexponentialcutoffpowerlaw( L_cut__Fermi, coeff, delta, nu, chi, z, z_binned__Fermi[j+1] )
		#~ Fermi_norm			=	obsFermi_long[j].sum() / model_bin__Fermi.sum()
		#~ print 'norm			:	', Fermi_norm
	#~ model_bin__Fermi	=	model_evolvingexponentialcutoffpowerlaw( L_cut__Fermi, coeff, delta, nu, chi, z, z_binned__Fermi[j+1] )
	#~ model_bin__Fermi	=	model_bin__Fermi * Fermi_norm
	#~ Fermi_long_total_model	=	Fermi_long_total_model + model_bin__Fermi
	#~ 
	#~ ax[j] = fig.add_subplot(z_binned__Fermi.size-1,1,j+1)
	#~ ax[j].text( -3.5, 350, r'$ z : $ ' + r'$ {0:.1f} $'.format(z) + r' $ \rm{to} $ ' + r'$ {0:.1f} $'.format(z_binned__Fermi[j+1]), fontsize = size_font, ha = 'center', va = 'center' )
	#~ ax[j].set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
	#~ ax[j].set_ylim( 0, 400 )
	#~ ax[j].errorbar( Luminosity_mids, obsFermi_long[j], yerr = [ obsFermi_long_poserr[j], obsFermi_long_negerr[j] ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
	#~ ax[j].plot( Luminosity_mids, model_bin__Fermi, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
	#~ ax[j].legend()
	#~ ltext = plt.gca().get_legend().get_texts()
	#~ plt.setp( ltext[0], fontsize = 11 )
	#~ 
	#~ modeled__Fermi_long[j]			=	model_bin__Fermi
	#~ discrepancy_over_z__Fermi[j]	=	find_discrepancy( model_bin__Fermi, obsFermi_long[j] )
	#~ rdcdchisqrd_over_z__Fermi[j]	=	mf.reduced_chisquared( model_bin__Fermi, obsFermi_long[j], obsFermi_long_error[j], 4 )[2]
	#~ 
	#~ #	print mf.reduced_chisquared( model_bin__Fermi, obsFermi_long[j], obsFermi_long_error[j], 4 )
#~ print 'Done in {:.3f} mins.'.format( ( time.time()-t0 )/60 )
#~ xticklabels	=	ax[0].get_xticklabels() + ax[1].get_xticklabels()
#~ plt.setp( xticklabels, visible = False )
#~ plt.xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font, labelpad = padding+5 )
#~ plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/tests--ECPL/Fermi--binned.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ Fermi_long_total_data_e	=	np.maximum(Fermi_long_total_data_ep, Fermi_long_total_data_en)+1
#~ print 'reduced chisquared	:	', mf.reduced_chisquared( Fermi_long_total_model, Fermi_long_total_data, Fermi_long_total_data_e, 4 ), '\n\n'
#~ ax	=	plt.subplot(111)
#~ ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
#~ ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
#~ ax.set_ylim( 0, 700 )
#~ ax.errorbar( Luminosity_mids, Fermi_long_total_data, yerr = [ Fermi_long_total_data_ep, Fermi_long_total_data_en ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
#~ ax.plot( Luminosity_mids, Fermi_long_total_model, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
#~ ax.legend()
#~ plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/tests--ECPL/Fermi--total.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ 
#~ 
#~ 
#~ coeff = 5.40	;	delta = 0.0	;	nu = 0.60	;	chi = 0.0
#~ 
#~ 
#~ print 'Swift...'
#~ Swift_long_L	=	known_long_Luminosity.copy()	;	Swift_long_L_error	=	Swift_long_Luminosity_error.copy()
#~ other_long_L	=	other_long_Luminosity.copy()	;	other_long_L_error	=	other_long_Luminosity_error.copy()
#~ Swift_long_total_data		=	np.zeros( Luminosity_mids.size )
#~ Swift_long_total_data_ep	=	np.zeros( Luminosity_mids.size )
#~ Swift_long_total_data_en	=	np.zeros( Luminosity_mids.size )
#~ Swift_long_total_model		=	np.zeros( Luminosity_mids.size )
#~ modeled__Swift_long			=	np.zeros( (z_binned__Swift.size-1 , Luminosity_mids.size) )
#~ discrepancy_over_z__Swift	=	np.zeros( z_binned__Swift.size )
#~ rdcdchisqrd_over_z__Swift	=	discrepancy_over_z__Swift.copy()
#~ 
#~ ax	=	{}
#~ fig	=	plt.figure( figsize=(6, 6) )
#~ fig.suptitle( r'$ Swift $', fontsize = size_font-2 )
#~ 
#~ t0	=	time.time()
#~ for j, z in enumerate( z_binned__Swift[:-1] ):
	#~ 
	#~ inds_to_take	=	np.where(  ( z < Swift_long_redshift ) & ( Swift_long_redshift < z_binned__Swift[j+1] )  )[0]
	#~ Swift_long_L	=	Swift_long_Luminosity[inds_to_take]	;	Swift_long_L_error	=	Swift_long_Luminosity_error[inds_to_take]
	#~ inds_to_take	=	np.where(  ( z < other_long_redshift ) & ( other_long_redshift < z_binned__Swift[j+1] )  )[0]
	#~ other_long_L	=	other_long_Luminosity[inds_to_take]	;	other_long_L_error	=	other_long_Luminosity_error[inds_to_take]
	#~ x__Swift_long_bin, y__Swift_long_bin, y__Swift_long_poserr_bin, y__Swift_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(Swift_long_L/L_norm), np.log10( (Swift_long_L + Swift_long_L_error) / L_norm ) - np.log10(Swift_long_L/L_norm), np.log10( (Swift_long_L + Swift_long_L) / L_norm ) - np.log10(Swift_long_L/L_norm), logL_bin, logL_min, logL_max )
	#~ x__other_long_bin, y__other_long_bin, y__other_long_poserr_bin, y__other_long_negerr_bin	=	sf.my_histogram_with_errorbars( np.log10(other_long_L/L_norm), np.log10( (other_long_L + other_long_L_error) / L_norm ) - np.log10(other_long_L/L_norm), np.log10( (other_long_L + other_long_L) / L_norm ) - np.log10(other_long_L/L_norm), logL_bin, logL_min, logL_max )
	#~ obsSwift_long_bin		=	y__Swift_long_bin + y__other_long_bin
	#~ obsSwift_long_poserr_bin=	np.sqrt(  y__Swift_long_poserr_bin**2 + y__other_long_poserr_bin**2  )
	#~ obsSwift_long_negerr_bin=	np.sqrt(  y__Swift_long_negerr_bin**2 + y__other_long_negerr_bin**2  )
	#~ obsSwift_long_error_bin	=	np.maximum(obsSwift_long_negerr_bin, obsSwift_long_poserr_bin)+1
	#~ Swift_long_total_data	=	Swift_long_total_data	+ obsSwift_long_bin
	#~ Swift_long_total_data_ep=	np.sqrt( Swift_long_total_data_ep**2 + obsSwift_long_poserr_bin**2 )
	#~ Swift_long_total_data_en=	np.sqrt( Swift_long_total_data_en**2 + obsSwift_long_negerr_bin**2 )
	#~ 
	#~ if j == 0:
		#~ model_bin__Swift	=	model_evolvingexponentialcutoffpowerlaw( L_cut__Swift, coeff, delta, nu, chi, z, z_binned__Swift[j+1] )
		#~ Swift_norm			=	obsSwift_long[j].sum() / model_bin__Swift.sum()
		#~ print 'norm			:	', Swift_norm
	#~ model_bin__Swift	=	model_evolvingexponentialcutoffpowerlaw( L_cut__Swift, coeff, delta, nu, chi, z, z_binned__Swift[j+1] )
	#~ model_bin__Swift	=	model_bin__Swift * Swift_norm
	#~ Swift_long_total_model	=	Swift_long_total_model + model_bin__Swift
	#~ 
	#~ ax[j] = fig.add_subplot(z_binned__Swift.size-1,1,j+1)
	#~ ax[j].text( -3.5, 140, r'$ z : $ ' + r'$ {0:.1f} $'.format(z) + r' $ \rm{to} $ ' + r'$ {0:.1f} $'.format(z_binned__Swift[j+1]), fontsize = size_font, ha = 'center', va = 'center' )
	#~ ax[j].set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
	#~ ax[j].set_ylim( 0, 160 )
	#~ ax[j].errorbar( Luminosity_mids, obsSwift_long[j], yerr = [ obsSwift_long_poserr[j], obsSwift_long_negerr[j] ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
	#~ ax[j].plot( Luminosity_mids, model_bin__Swift, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
	#~ ax[j].legend()
	#~ ltext = plt.gca().get_legend().get_texts()
	#~ plt.setp( ltext[0], fontsize = 11 )
	#~ 
	#~ modeled__Swift_long[j]			=	model_bin__Swift
	#~ discrepancy_over_z__Swift[j]	=	find_discrepancy( model_bin__Swift, obsSwift_long[j] )
	#~ rdcdchisqrd_over_z__Swift[j]	=	mf.reduced_chisquared( model_bin__Swift, obsSwift_long[j], obsSwift_long_error[j], 4 )[2]
				#~ 
	#~ #	print mf.reduced_chisquared( model_bin__Swift, obsSwift_long[j], obsSwift_long_error[j], 4 )
#~ print 'Done in {:.3f} mins.'.format( ( time.time()-t0 )/60 )
#~ xticklabels	=	ax[0].get_xticklabels() + ax[1].get_xticklabels()
#~ plt.setp( xticklabels, visible = False )
#~ plt.xlabel( r'$ \rm{ log } $' + r'$ ( L_{iso} / L_{0} ) $', fontsize = size_font, labelpad = padding+5 )
#~ plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/tests--ECPL/Swift--binned.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ Swift_long_total_data_e	=	np.maximum(Swift_long_total_data_ep, Swift_long_total_data_en)+1
#~ print 'reduced chisquared	:	', mf.reduced_chisquared( Swift_long_total_model, Swift_long_total_data, Swift_long_total_data_e, 4 ), '\n\n'
#~ ax	=	plt.subplot(111)
#~ ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
#~ ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
#~ ax.set_ylim( 0, 250 )
#~ ax.errorbar( Luminosity_mids, Swift_long_total_data, yerr = [ Swift_long_total_data_ep, Swift_long_total_data_en ], fmt = '-', color = 'k', label = r' $ \rm{ observed } $' )
#~ ax.plot( Luminosity_mids, Swift_long_total_model, linestyle = '--', color = 'k', label = r' $ \rm{ model } $' )
#~ ax.legend()
#~ plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/tests--ECPL/Swift--total.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ 
#~ long_total_data		=	Fermi_long_total_data	+	Swift_long_total_data
#~ long_total_data_ep	=	np.sqrt( Fermi_long_total_data_ep**2 + Swift_long_total_data_ep**2 )
#~ long_total_data_en	=	np.sqrt( Fermi_long_total_data_en**2 + Swift_long_total_data_en**2 )
#~ long_total_data_e	=	np.maximum(long_total_data_ep, long_total_data_en)+1
#~ long_total_model	=	Fermi_long_total_model	+	Swift_long_total_model
#~ print mf.reduced_chisquared( long_total_model, long_total_data, long_total_data_e, 4 ), '\n\n'
#~ #	print np.sum( Fermi_long_total_data ), np.sum( Swift_long_total_data )
#~ 
#~ 
#~ 
#~ ####################################################################################################################################################





print 'CZTI...'
CZTI_long_total_model		=	np.zeros( Luminosity_mids.size )
ax	=	{}
fig	=	plt.figure( figsize=(6,6) )
#~ fig.suptitle( r'$ \rm{ CZTI } $', fontsize = size_font-2 )
for j, z in enumerate( z_binned__Fermi[:-1] ):
	
	model_bin__CZTI	=	model_evolvingexponentialcutoffpowerlaw( L_cut__Fermi, coeff, delta, nu, chi, z, z_binned__Fermi[j+1] )
	model_bin__CZTI	=	model_bin__CZTI * ( 12.329*1e-8 / 3 )
	
	print round( model_bin__CZTI.sum() )
	
	CZTI_long_total_model	=	CZTI_long_total_model + model_bin__CZTI
	
	ax[j] = fig.add_subplot(z_binned__Fermi.size-1,1,j+1)
	ax[j].text( -3.2, 22, r'$ z : $ ' + r'$ {0:.1f} $'.format(z) + r' $ \rm{to} $ ' + r'$ {0:.1f} $'.format(z_binned__Fermi[j+1]), fontsize = size_font, ha = 'center', va = 'center' )
	ax[j].set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
	ax[j].set_ylim( 0, 30 )
	ax[j].plot( Luminosity_mids, model_bin__CZTI, linestyle = '--', color = 'k' )
xticklabels	=	ax[0].get_xticklabels() + ax[1].get_xticklabels()
plt.setp( xticklabels, visible = False )
plt.xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding+5 )
plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/CZTI--binned.png' )
plt.clf()
plt.close()


ax	=	plt.subplot(111)
ax.set_xlabel( r'$ \rm{ log } $' + r'$ ( L_{p} / L_{0} ) $', fontsize = size_font, labelpad = padding-4 )
ax.set_ylabel( r'$ \rm{ N } $', fontsize = size_font, rotation = 0, labelpad = padding+6 )
ax.set_ylim( 0, 50 )
ax.plot( Luminosity_mids, CZTI_long_total_model, linestyle = '--', color = 'k' )
plt.savefig( './../plots/estimated_lumfunc_models/binned_with_z/CZTI--total.png' )
plt.clf()
plt.close()

print 'Total number of predicted CZTI GRBs:	', round( CZTI_long_total_model.sum() )

