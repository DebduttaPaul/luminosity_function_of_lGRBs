from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.stats import pearsonr as R
from scipy.stats import spearmanr as S
from scipy.stats import kendalltau as T
from scipy.optimize import curve_fit
from scipy.integrate import quad
import debduttaS_functions as mf
import specific_functions as sf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
#~ plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('font', family = 'serif', serif = 'cm10', size = 15)
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	7	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	1e+1 #	for the purposes of plotting

P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6022 * 1e-9

A___Yonetoku	=	23.4		#	Yonetoku correlation
eta_Yonetoku	=	2.0			#	Yonetoku correlation
A___Tanbestfit	=	3.47		#	best-fit Lp-Ep, Tan-2013
eta_Tanbestfit	=	1.28		#	best-fit Lp-Ep, Tan-2013
A___Tanredshift	=	7.93		#	best-fit redshift-distribution, Tan-2013
eta_Tanredshift	=	1.70		#	best-fit redshift-distribution, Tan-2013


x_in_keV_min	=	1e00	;	x_in_keV_max	=	2e04
y_in_eps_min	=	1e49	;	y_in_eps_max	=	1e56


####################################################################################################################################################



####################################################################################################################################################


def straight_line( x, m, c ):
	return m*x + c


####################################################################################################################################################



####################################################################################################################################################


Fermi_GRBs_table			=	ascii.read( './../tables/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_name					=	Fermi_GRBs_table['Fermi name'].data
Fermi_Epeak					=	Fermi_GRBs_table['Epeak'].data
Fermi_Epeak_error			=	Fermi_GRBs_table['Epeak_error'].data
Fermi_alpha					=	Fermi_GRBs_table['alpha'].data
Fermi_alpha_error			=	Fermi_GRBs_table['alpha_error'].data
Fermi_beta					=	Fermi_GRBs_table['beta'].data
Fermi_beta_error			=	Fermi_GRBs_table['beta_error'].data

common_GRBs_table			=	ascii.read( './../tables/common_GRBs--wkr.txt', format = 'fixed_width' )
common_ID					=	common_GRBs_table['common ID'].data
common_Fermi_name			=	common_GRBs_table['Fermi name'].data
common_Swift_T90			=	common_GRBs_table['BAT T90'].data
common_Fermi_T90			=	common_GRBs_table['GBM T90'].data
common_Fermi_T90_error		=	common_GRBs_table['GBM T90_error'].data
common_redshift				=	common_GRBs_table['redshift'].data
common_Fermi_flux			=	common_GRBs_table['GBM flux'].data
common_Fermi_flux_error		=	common_GRBs_table['GBM flux_error'].data
common_Epeak				=	common_GRBs_table['Epeak'].data				#	in keV.
common_Epeak_error			=	common_GRBs_table['Epeak_error'].data		#	in keV.
common_alpha				=	common_GRBs_table['alpha'].data
common_alpha_error			=	common_GRBs_table['alpha_error'].data
common_beta					=	common_GRBs_table['beta'].data
common_beta_error			=	common_GRBs_table['beta_error'].data
common_Luminosity			=	common_GRBs_table['Luminosity'].data
common_Luminosity_error		=	common_GRBs_table['Luminosity_error'].data
common_num					=	common_ID.size


####################################################################################################################################################





####################################################################################################################################################

##	To extract the average parameters of the Fermi GRBs.

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_Epeak), 0.125, 1, 4 )	;	hxF	=	hist[0]	;	hyF	=	hist[1]
fits	=	mf.fit_a_gaussian( hxF, hyF )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]
print 'Fermi spectral parameters -- statistics :'
print 'Epeak					:	', np.median(Fermi_Epeak), mf.std_via_mad(Fermi_Epeak)
print 'alpha					:	', np.median(Fermi_alpha), mf.std_via_mad(Fermi_alpha)
print 'beta 					:	', np.median(Fermi_beta ), mf.std_via_mad(Fermi_beta )
print 'Epeak, fit				:	', 10**f0, 10**f1
print '\n\n\n\n'


####################################################################################################################################################





####################################################################################################################################################


common_Epeak_in_keV			=	common_Epeak								# in keV.
common_Epeak_in_keV_error	=	common_Epeak_error							# same as above.
common_Epeak_in_MeV			=	1e-3 * common_Epeak_in_keV					# in MeV
common_Epeak_in_MeV_error	=	1e-3 * common_Epeak_in_keV_error			# same as above.
x_to_fit_in_keV				=	common_Epeak_in_keV*(1+common_redshift)		# in keV.
x_to_fit_in_keV_error		=	common_Epeak_in_keV_error					# same as above.
x_to_fit_in_MeV				=	common_Epeak_in_MeV*(1+common_redshift)		# in MeV.
x_to_fit_in_MeV_error		=	common_Epeak_in_MeV_error					# same as above.
y_to_fit					=	common_Luminosity							# in units of L_norm.
y_to_fit_error				=	common_Luminosity_error						# same as above.
common_Luminosity			=	L_norm * common_Luminosity					# in erg.s^{-1}.
common_Luminosity_error		=	L_norm * common_Luminosity_error			# same as above.



#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( common_redshift, common_Luminosity, yerr = common_Luminosity_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
#~ plt.savefig( './../plots/L_vs_z--common_GRBs__wkr--all.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ E_p \, (1+z) \; \rm{ [keV] } $', fontsize = size_font )
#~ ax.set_ylabel( r'$ L_p \; \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_xlim( x_in_keV_min, x_in_keV_max )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( x_to_fit_in_keV, common_Luminosity, xerr = x_to_fit_in_keV_error, yerr = common_Luminosity_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
#~ plt.savefig( './../plots/L_vs_Ep(1+z)--common_GRBs__wkr--all.png' )
#~ plt.clf()
#~ plt.close()


ind_short_Fermi		=	np.where( (common_Fermi_T90 < T90_cut) )[0]
ind_short_Swift		=	np.where( (common_Swift_T90 < T90_cut) )[0]
ind_short_both		=	np.intersect1d( ind_short_Fermi, ind_short_Swift )
ind_short_either	=	np.unique( np.union1d(ind_short_Fermi, ind_short_Swift) )
ind_long_both		=	np.delete( np.arange(common_num), ind_short_either )
print '\n\n'
print 'total number	:	', common_redshift.size
print 'long in both		:	', ind_long_both.size, ind_long_both
print 'short in Fermi 		:	', ind_short_Fermi.size, ind_short_Fermi
print 'short in Swift 		:	', ind_short_Swift.size, ind_short_Swift
print 'short in both		:	', ind_short_both.size , ind_short_both
ind_short_Fermi_only	=	mf.delete( ind_short_Fermi, ind_short_both )
ind_short_Swift_only	=	mf.delete( ind_short_Swift, ind_short_both )
print 'short in Fermi only	:	', ind_short_Fermi_only.size, ind_short_Fermi_only, common_Fermi_T90[ind_short_Fermi_only], common_Fermi_T90_error[ind_short_Fermi_only], common_Swift_T90[ind_short_Fermi_only]
print 'short in Swift only	:	', ind_short_Swift_only.size, ind_short_Swift_only, common_Fermi_T90[ind_short_Swift_only], common_Fermi_T90_error[ind_short_Swift_only], common_Swift_T90[ind_short_Swift_only]
ind_long	=	np.sort( np.append(ind_long_both, ind_short_Fermi_only) )
ind_short	=	ind_short_both.copy()
print 'long			:	', ind_long.size , ind_long
print 'short			:	', ind_short.size, ind_short
print '\n\n'


#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ L_p \; \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( common_redshift[ind_short], common_Luminosity[ind_short], yerr = common_Luminosity_error[ind_short], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'r', markeredgecolor = 'r', label = r'$ \rm{short} $' )
#~ ax.errorbar( common_redshift[ind_long ], common_Luminosity[ind_long ], yerr = common_Luminosity_error[ind_long ], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k', label = r'$ \rm{ long} $' )
#~ plt.legend( numpoints = 1, loc = 'upper left' )
#~ plt.savefig( './../plots/L_vs_z--common_GRBs__wkr--T90.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ E_p \, (1+z) \; \rm{ [keV] } $', fontsize = size_font )
#~ ax.set_ylabel( r'$ L_p \; \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_xlim( x_in_keV_min, x_in_keV_max )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( x_to_fit_in_keV[ind_short], common_Luminosity[ind_short], xerr = x_to_fit_in_keV_error[ind_short], yerr = common_Luminosity_error[ind_short], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'r', markeredgecolor = 'r', label = r'$ \rm{short} $' )
#~ ax.errorbar( x_to_fit_in_keV[ind_long ], common_Luminosity[ind_long ], xerr = x_to_fit_in_keV_error[ind_long ], yerr = common_Luminosity_error[ind_long ], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k', label = r'$ \rm{ long} $' )
#~ plt.legend( numpoints = 1, loc = 'upper left' )
#~ plt.savefig( './../plots/L_vs_Ep(1+z)--common_GRBs__wkr--T90.png' )
#~ plt.clf()
#~ plt.close()


####################################################################################################################################################




####################################################################################################################################################

##	To check for correlations between the source quantities L_p and E_peak.
print '\n\n\n\n'
print 'Correlations...', '\n'
print 'All:	'
r, p_r	=	R( x_to_fit_in_MeV, y_to_fit )
s, p_s	=	S( x_to_fit_in_MeV, y_to_fit )
t, p_t	=	T( x_to_fit_in_MeV, y_to_fit )
print r, p_r
print s, p_s
print t, p_t
print '\n',
print 'Short:	'
r, p_r	=	R( x_to_fit_in_MeV[ind_short], y_to_fit[ind_short] )
s, p_s	=	S( x_to_fit_in_MeV[ind_short], y_to_fit[ind_short] )
t, p_t	=	T( x_to_fit_in_MeV[ind_short], y_to_fit[ind_short] )
print r, p_r
print s, p_s
print t, p_t
print '\n',
print 'Long:	'
r, p_r	=	R( x_to_fit_in_MeV[ind_long], y_to_fit[ind_long] )
s, p_s	=	S( x_to_fit_in_MeV[ind_long], y_to_fit[ind_long] )
t, p_t	=	T( x_to_fit_in_MeV[ind_long], y_to_fit[ind_long] )
print r, p_r
print s, p_s
print t, p_t
print '\n\n\n\n'

##	To choose only the GRBs that are long, for the correlation-studies.
inds						=	ind_long
x_to_fit_in_keV				=	x_to_fit_in_keV[inds]
x_to_fit_in_keV_error		=	x_to_fit_in_keV_error[inds]
x_to_fit_in_MeV				=	x_to_fit_in_MeV[inds]
x_to_fit_in_MeV_error		=	x_to_fit_in_MeV_error[inds]
y_to_fit					=	y_to_fit[inds]
y_to_fit_error				=	y_to_fit_error[inds]
common_Luminosity			=	common_Luminosity[inds]
common_Luminosity_error		=	common_Luminosity_error[inds]

common_Fermi_name			=	common_Fermi_name[inds]
common_Fermi_flux			=	common_Fermi_flux[inds]
common_Fermi_flux_error		=	common_Fermi_flux_error[inds]
common_Epeak_in_keV			=	common_Epeak_in_keV[inds]
common_Epeak_in_keV_error	=	common_Epeak_in_keV_error[inds]
common_Epeak_in_MeV			=	common_Epeak_in_MeV[inds]
common_Epeak_in_MeV_error	=	common_Epeak_in_MeV_error[inds]
common_redshift				=	common_redshift[inds]
common_num					=	inds.size

print '\n',
r, p_r	=	R( x_to_fit_in_MeV, y_to_fit )
s, p_s	=	S( x_to_fit_in_MeV, y_to_fit )
t, p_t	=	T( x_to_fit_in_MeV, y_to_fit )
print r, p_r
print s, p_s
print t, p_t
print '\n\n\n\n'


##	To extract the present best-fit.
x_to_fit_log	=	np.log10(x_to_fit_in_MeV)
y_to_fit_log	=	np.log10(y_to_fit)
popt, pcov = curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
eta_mybestfit	=	popt[0]	;	A___mybestfit	=	10**popt[1]
print '\n\n\n\n'
print 'My best-fit...', '\n'
print 'A   mean, error:	', round( A___mybestfit, 3 ), round( (10**pcov[1,1]-1)*A___mybestfit, 3 )
print 'eta mean, error:	', round( eta_mybestfit, 3 ), round( pcov[0,0], 3 )
print '\n\n\n\n'

#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ E_p \, (1+z) \; \rm{ [keV] } $', fontsize = size_font )
#~ ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_xlim( x_in_keV_min, x_in_keV_max )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( x_to_fit_in_keV, common_Luminosity, xerr = x_to_fit_in_keV_error, yerr = common_Luminosity_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Yonetoku    )  * ( x_to_fit_in_MeV**eta_Yonetoku    ), color = 'b', label = r'$ \rm{ Yonetoku \, (2004) : best \, fit } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Tanbestfit  )  * ( x_to_fit_in_MeV**eta_Tanbestfit  ), color = 'r', label = r'$ \rm{ Tan \, (2013) : best \, fit } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Tanredshift )  * ( x_to_fit_in_MeV**eta_Tanredshift ), color = 'y', label = r'$ \rm{ Tan \, (2013) : new \, method  } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___mybestfit   )  * ( x_to_fit_in_MeV**eta_mybestfit   ), color = 'g', label = r'$ \rm{ present \, work : best \, fit } $' )
#~ plt.legend( numpoints = 1, loc = 'upper left' )
#~ plt.savefig( './../plots/L_vs_Ep(1+z)--correlations--my_bestfit.png' )
#~ plt.clf()
#~ plt.close()


####################################################################################################################################################






#~ ####################################################################################################################################################
#~ 
#~ 
#~ k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
#~ z_sim		=	k_table['z'].data
#~ dL_sim		=	k_table['dL'].data
#~ k_Fermi		=	k_table['k_Fermi'].data
#~ k_Swift		=	k_table['k_Swift'].data
#~ term_Fermi	=	k_table['term_Fermi'].data
#~ z_bin		=	np.mean( np.diff(z_sim) )
#~ 
#~ numerator__delta_pseudo_GRB__first_term		=	common_Fermi_flux_error / common_Fermi_flux											#	defined for each GRB
#~ numerator__delta_pseudo_GRB__second_term	=	common_Epeak_in_MeV_error / common_Epeak_in_MeV										#	defined for each GRB
#~ denominator__delta_pseudo_zsim__first_term	=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )   + term_Fermi		#	defined for each simulated redshift
#~ 
#~ #	A_min	=	1e-3	;	A_max	=	40.00	;	A_bin	=	1.00	#1
#~ #	eta_min	=	1e-3	;	eta_max	=	 4.00	;	eta_bin	=	0.50	#1
#~ #	A_min	=	1.00	;	A_max	=	 10.00	;	A_bin	=	0.50	#2
#~ #	eta_min	=	1e-3	;	eta_max	=	 2.00	;	eta_bin	=	0.10	#2
#~ #	A_min	=	3.00	;	A_max	=	 20.00	;	A_bin	=	0.50	#3
#~ #	eta_min	=	1e-3	;	eta_max	=	 2.00	;	eta_bin	=	0.10	#3
#~ A_min	=	3.00	;	A_max	=	 40.00	;	A_bin	=	0.50	#4
#~ eta_min	=	1e-3	;	eta_max	=	 3.00	;	eta_bin	=	0.10	#4
#~ 
#~ eta_array	=	np.arange( eta_min, eta_max, eta_bin )
#~ A_array		=	np.arange(   A_min,   A_max,   A_bin )
#~ grid_of_discrepancy	=	np.zeros( (eta_array.size, A_array.size) )
#~ grid_of_redchisqrd	=	np.zeros( (eta_array.size, A_array.size) )
#~ print '\n\n\n'
#~ print 'Grid of  {0:d} (A)  X  {1:d} (eta)  =  {2:d}'.format(A_array.size, eta_array.size, grid_of_discrepancy.size)
#~ 
#~ t0	=	time.time()
#~ 
#~ numerator__F		=	k_Fermi  *  4*P  *  ( dL_sim**2 )
#~ for ceta, eta in enumerate(eta_array):
	#~ denominator__F	=	(1+z_sim)**eta
	#~ F_Fermi_sim		=	numerator__F/denominator__F
	#~ F_Fermi_sim		=	F_Fermi_sim * (cm_per_Mpc**2) / L_norm
	#~ 
	#~ numerator__delta_pseudo_z		=	numerator__delta_pseudo_GRB__first_term  +  eta * numerator__delta_pseudo_GRB__second_term		#	defined for each GRB
	#~ denominator__delta_pseudo_z		=	denominator__delta_pseudo_zsim__first_term  +  (  (eta+2) / (1+z_sim)  )						#	defined for each simulated redshift	
	#~ for cA, A in enumerate(A_array):
		#~ 
		#~ ##	To directly use the assumed relationship:	pseudo_zs		=	( (common_Luminosities/(A*L_norm))**(1/eta) ) / common_Epeaks  -  1
		#~ 
		#~ RHSs	=	( A / common_Fermi_flux ) * ( common_Epeak_in_MeV**eta )
		#~ pseudo_redshifts		=	np.zeros( RHSs.size )
		#~ pseudo_redshifts_error	=	np.zeros( RHSs.size )
		#~ for j, RHS in enumerate(RHSs):
			#~ 
			#~ array	=	np.abs( F_Fermi_sim - RHS )
			#~ ind		=	np.where(  array == array.min()  )[0][0]
			#~ pseudo_redshift			=	z_sim[ind]
			#~ pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z[ind]
			#~ 
			#~ pseudo_redshifts[j]			=	pseudo_redshift
			#~ pseudo_redshifts_error[j]	=	pseudo_redshift_error
		#~ 
		#~ z_min		=	1e-5	;	z_max		=	1e1
		#~ x1, y_pseudo	=	mf.my_histogram_according_to_given_boundaries( pseudo_redshifts, z_bin, z_min, z_max )
		#~ x2, y_common	=	mf.my_histogram_according_to_given_boundaries( common_redshift , z_bin, z_min, z_max )
		#~ #	print ( x1 - x2 == 0 ).all()
		#~ discrepancy	=	np.sum( (y_pseudo - y_common)**2 )
		#~ 
		#~ redchisqrd	=	mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )[2]
		#~ 
		#~ grid_of_discrepancy[ceta, cA]	=	discrepancy
		#~ grid_of_redchisqrd[ ceta, cA]	=	redchisqrd
#~ 
#~ 
#~ ind_discrepancy_min	=	np.unravel_index( grid_of_discrepancy.argmin(), grid_of_discrepancy.shape )
#~ print 'Minimum of {0:.3f} at A = {1:.2f} and eta = {2:.2f}.'.format( grid_of_discrepancy[ind_discrepancy_min], A_array[ind_discrepancy_min[1]], eta_array[ind_discrepancy_min[0]] )
#~ 
#~ 
#~ plt.contourf( A_array, eta_array, grid_of_discrepancy )
#~ plt.colorbar()
#~ plt.xlabel( r'$ \mathrm{ A } $', fontsize = size_font + 2, labelpad = padding - 3 )
#~ plt.ylabel( r'$ \eta $', fontsize = size_font + 2, labelpad = padding )
#~ plt.savefig( './../plots/Tan_method--blind/discrepancy_contours.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ levels = range( 0, 200, 1 )										#4
#~ plt.contourf( A_array, eta_array, grid_of_redchisqrd, levels )	#4
#~ #	plt.contourf( A_array, eta_array, grid_of_redchisqrd )	#1,2,3
#~ plt.colorbar()
#~ plt.xlabel( r'$ \mathrm{ A } $', fontsize = size_font + 2, labelpad = padding - 3 )
#~ plt.ylabel( r'$ \eta $', fontsize = size_font + 2, labelpad = padding )
#~ plt.savefig( './../plots/Tan_method--blind/redchisqrd_contours.png' )
#~ plt.clf()
#~ plt.close()
#~ 
#~ print 'Done in {:.3f} mins.'.format( ( time.time()-t0 )/60 ), '\n\n'
#~ 
#~ 
#~ ####################################################################################################################################################








####################################################################################################################################################


k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
term_Fermi	=	k_table['term_Fermi'].data
z_bin		=	np.mean( np.diff(z_sim) )

numerator__delta_pseudo_GRB__first_term		=	common_Fermi_flux_error / common_Fermi_flux											#	defined for each GRB
numerator__delta_pseudo_GRB__second_term	=	common_Epeak_in_MeV_error / common_Epeak_in_MeV										#	defined for each GRB
denominator__delta_pseudo_zsim__first_term	=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )   + term_Fermi		#	defined for each simulated redshift

numerator__F								=	k_Fermi  *  4*P  *  ( dL_sim**2 )
def discrepancy_and_redchisqrd( A, eta ):
	
	numerator__delta_pseudo_z		=	numerator__delta_pseudo_GRB__first_term  +  eta * numerator__delta_pseudo_GRB__second_term		#	defined for each GRB
	denominator__delta_pseudo_z		=	denominator__delta_pseudo_zsim__first_term  +  (  (eta+2) / (1+z_sim)  )						#	defined for each simulated redshift
	
	denominator__F	=	(1+z_sim)**eta
	F_Fermi_sim		=	numerator__F/denominator__F
	F_Fermi_sim		=	F_Fermi_sim * (cm_per_Mpc**2) / L_norm
	RHSs			=	( A / common_Fermi_flux ) * ( common_Epeak_in_MeV**eta )
	
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Fermi_sim - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	z_min		=	1e-5	;	z_max		=	1e1
	x1, y_pseudo	=	mf.my_histogram_according_to_given_boundaries( pseudo_redshifts, z_bin, z_min, z_max )
	x2, y_common	=	mf.my_histogram_according_to_given_boundaries( common_redshift , z_bin, z_min, z_max )
	#	print ( x1 - x2 == 0 ).all()
	discrepancy	=	np.sum( (y_pseudo - y_common)**2 )
	
	chisquared, dof, redchisqrd	=	mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
	
	
	return pseudo_redshifts, pseudo_redshifts_error, discrepancy, chisquared, dof, redchisqrd


def plot_distribution( pseudo_redshifts ):
	
	z_min		=	1e-5
	z_max		=	1e1
	z_bin		=	1e0
	width		=	1.0
	ymin		=	00
	ymax		=	30
	x, y	=	mf.my_histogram_according_to_given_boundaries( pseudo_redshifts, z_bin, z_min, z_max )
	plt.ylim( ymin, ymax )
	plt.bar( x, y, width = width, color = 'w', edgecolor = 'k', hatch = '//', label = r'$ \rm{ pseudo } $' )
	x, y	=	mf.my_histogram_according_to_given_boundaries( common_redshift , z_bin, z_min, z_max )
	plt.bar( x, y, width = width, color = 'c', alpha = 0.5, label = r'$ \rm{ observed } $' )
	plt.xlabel( r'$ \rm {redshift} $', fontsize = size_font+2 )
	plt.legend()


returned	=	discrepancy_and_redchisqrd( A___Yonetoku   , eta_Yonetoku    )
print 'Yonetoku:	', returned[2], returned[3], returned[4], returned[5]
plot_distribution( returned[0] )
plt.savefig( './../plots/Tan_method--blind/distribution--Yonetoku.png' )
plt.clf()
plt.close()

returned	=	discrepancy_and_redchisqrd( A___Tanbestfit , eta_Tanbestfit  )
print 'Tan: best-fit:	', returned[2], returned[3], returned[4], returned[5]
plot_distribution( returned[0] )
plt.savefig( './../plots/Tan_method--blind/distribution--Tan_bestfit.png' )
plt.clf()
plt.close()

returned	=	discrepancy_and_redchisqrd( A___Tanredshift, eta_Tanredshift )
print 'Tan, redshift:	', returned[2], returned[3], returned[4], returned[5]
plot_distribution( returned[0] )
plt.savefig( './../plots/Tan_method--blind/distribution--Tan_redshift.png' )
plt.clf()
plt.close()

returned	=	discrepancy_and_redchisqrd( A___mybestfit  , eta_mybestfit   )
print 'my  best-fit:	', returned[2], returned[3], returned[4], returned[5]
plot_distribution( returned[0] )
plt.savefig( './../plots/Tan_method--blind/distribution--my_bestfit.png' )
plt.clf()
plt.close()
table_of_redshifts	=	Table( [common_Fermi_name, common_redshift, returned[0], returned[1], x_to_fit_in_keV, x_to_fit_in_keV_error, common_Luminosity, common_Luminosity_error], names = ['Fermi name', 'measured', 'pseudo', 'pseudo_error', 'Ep(1+z) [keV]', 'Ep(1+z)_error [keV]', 'Luminosity', 'Luminosity_error'] )
ascii.write( table_of_redshifts, './../tables/redshifts--my_bestfit.txt', format = 'fixed_width', overwrite = True )


print '\n', 'Correlations between measured and pseudo redshifts...'
r, p_r	=	R( common_redshift, returned[0] )
s, p_s	=	S( common_redshift, returned[0] )
t, p_t	=	T( common_redshift, returned[0] )
print r, p_r
print s, p_s
print t, p_t
print '\n'


print common_redshift.size, returned[0].size
z_meas__sim	=	np.arange( 0.2, 10.0, 0.1 )
z_pseu__sim	=	z_meas__sim
ax = plt.subplot(111)
ax.set_xlim(0.1, 20.0)
ax.set_ylim(0.1, 20.0)
#	ax.set_title( r'$ p = 7 \times 10^{-3} $' )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \;\, redshift } $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ pseudo   \;\, redshift } $', fontsize = size_font+2, labelpad = padding-8 )
#~ ax.plot( common_redshift, returned[0], 'ko' )
ax.errorbar( common_redshift, returned[0], yerr = returned[1], fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
ax.plot( z_meas__sim, z_pseu__sim, color = 'k' )
plt.savefig('./../plots/Tan_method--blind/redshift_comparison.png')
plt.clf()
plt.close()
ratio	=	returned[0] / common_redshift
print '\n\n', 'pseudo : measured ::	'
print 'mean, median	:	' , np.mean( ratio ), np.median( ratio )
print 'std, mad 	:	', np.std( ratio ), mf.std_via_mad( ratio )


#~ ax = plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ E_p \, (1+z) \; \rm{ [keV] } $', fontsize = size_font )
#~ ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.set_xlim( x_in_keV_min, x_in_keV_max )
#~ ax.set_ylim( y_in_eps_min, y_in_eps_max )
#~ ax.errorbar( x_to_fit_in_keV, common_Luminosity, xerr = x_to_fit_in_keV_error, yerr = common_Luminosity_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Yonetoku    )  * ( x_to_fit_in_MeV**eta_Yonetoku    ), color = 'b'         , label = r'$ \rm{ Yonetoku \, (2004) : best \, fit } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Tanbestfit  )  * ( x_to_fit_in_MeV**eta_Tanbestfit  ), color = 'r'         , label = r'$ \rm{ Tan \, (2013) : best \, fit } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___Tanredshift )  * ( x_to_fit_in_MeV**eta_Tanredshift ), color = 'y'         , label = r'$ \rm{ Tan \, (2013) : new \, method  } $' )
#~ ax.plot( x_to_fit_in_keV, ( L_norm * A___mybestfit   )  * ( x_to_fit_in_MeV**eta_mybestfit   ), color = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
#~ plt.legend( numpoints = 1, loc = 'upper left' )
#~ plt.savefig( './../plots/Tan_method--blind/L_vs_Ep(1+z)--correlations--Tan_method_blind.png' )
#~ plt.clf()
#~ plt.close()


####################################################################################################################################################

