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
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	07	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	2e+1 #	for the purposes of plotting

P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.

L_norm		=	1e52	# in ergs.s^{-1}.
T90_cut		=	2		# in sec.

cm_per_Mpc	=	3.0857 * 1e24

A___Yonetoku	=	23.4		#	Yonetoku correlation
eta_Yonetoku	=	2.0			#	Yonetoku correlation
A___Tanbestfit	=	3.47		#	best-fit Lp-Ep, Tan-2013
eta_Tanbestfit	=	1.28		#	best-fit Lp-Ep, Tan-2013
A___Tanredshift	=	7.93		#	best-fit redshift-distribution, Tan-2013
eta_Tanredshift	=	1.70		#	best-fit redshift-distribution, Tan-2013
A___mybestfit	=	4.780		#	my best-fit
eta_mybestfit	=	1.229		#	my best-fit
A___myredshift	=	2.25		#	via pseudo redshift estimation
eta_myredshift	=	0.30		#	via pseudo redshift estimation


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



ind_short_Fermi		=	np.where( (common_Fermi_T90 < T90_cut) )[0]
ind_short_Swift		=	np.where( (common_Swift_T90 < T90_cut) )[0]
ind_short_both		=	np.intersect1d( ind_short_Fermi, ind_short_Swift )
ind_short_either	=	np.unique( np.union1d(ind_short_Fermi, ind_short_Swift) )
ind_long_both		=	np.delete( np.arange(common_num), ind_short_either )
print '\n\n'
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


##	To choose only the GRBs that are long in both for the correlation-studies.
inds						=	ind_long
x_to_fit_in_keV				=	x_to_fit_in_keV[inds]
x_to_fit_in_keV_error		=	x_to_fit_in_keV_error[inds]
x_to_fit_in_MeV				=	x_to_fit_in_MeV[inds]
x_to_fit_in_MeV_error		=	x_to_fit_in_MeV_error[inds]
y_to_fit					=	y_to_fit[inds]
y_to_fit_error				=	y_to_fit_error[inds]
common_Luminosity			=	common_Luminosity[inds]
common_Luminosity_error		=	common_Luminosity_error[inds]

common_Fermi_flux			=	common_Fermi_flux[inds]
common_Fermi_flux_error		=	common_Fermi_flux_error[inds]
common_Epeak_in_keV			=	common_Epeak_in_keV[inds]
common_Epeak_in_keV_error	=	common_Epeak_in_keV_error[inds]
common_Epeak_in_MeV			=	common_Epeak_in_MeV[inds]
common_Epeak_in_MeV_error	=	common_Epeak_in_MeV_error[inds]
common_redshift				=	common_redshift[inds]
common_num					=	inds.size


####################################################################################################################################################









####################################################################################################################################################


##	To construct the possible factor, let's call it M : predicted / observed (Luminosity).
possible_factor___Yonetoku		=	A___Yonetoku    *(x_to_fit_in_MeV**eta_Yonetoku   ) / y_to_fit
possible_factor___Tanbestfit 	=	A___Tanbestfit  *(x_to_fit_in_MeV**eta_Tanbestfit ) / y_to_fit
possible_factor___Tanredshift	=	A___Tanredshift *(x_to_fit_in_MeV**eta_Tanredshift) / y_to_fit
possible_factor___mybestfit 	=	A___mybestfit   *(x_to_fit_in_MeV**eta_mybestfit  ) / y_to_fit
possible_factor___myredshift 	=	A___myredshift  *(x_to_fit_in_MeV**eta_myredshift ) / y_to_fit

possible_factor___mybestfit_error	=	possible_factor___mybestfit   * (   ( eta_mybestfit  *(x_to_fit_in_keV_error/x_to_fit_in_keV) )  +  (y_to_fit_error/y_to_fit_error)   ) 
possible_factor___myredshift_error	=	possible_factor___myredshift  * (   ( eta_myredshift *(x_to_fit_in_keV_error/x_to_fit_in_keV) )  +  (y_to_fit_error/y_to_fit_error)   ) 


def check_for_correlations( against_what ):
	
	print 'Yonetoku...'
	high_num	=	np.where( possible_factor___Yonetoku / y_to_fit > 1 )[0].size
	print 'Number of points with M greater and lesser than one:	{0:d} and {1:d}'.format( high_num, common_num - high_num ), '\n'
	r, p_r	=	R( against_what[0], possible_factor___Yonetoku )
	s, p_s	=	S( against_what[0], possible_factor___Yonetoku )
	t, p_t	=	T( against_what[0], possible_factor___Yonetoku )
	print 'R, p:	', r, p_r
	print 'S, p:	', s, p_s
	print 'T, p:	', t, p_t
	print '\n',
	
	print 'Tan best-fit...'
	high_num	=	np.where( possible_factor___Tanbestfit / y_to_fit > 1 )[0].size
	print 'Number of points with M greater and lesser than one:	{0:d} and {1:d}'.format( high_num, common_num - high_num ), '\n'
	r, p_r	=	R( against_what[1], possible_factor___Tanbestfit )
	s, p_s	=	S( against_what[1], possible_factor___Tanbestfit )
	t, p_t	=	T( against_what[1], possible_factor___Tanbestfit )
	print 'R, p:	', r, p_r
	print 'S, p:	', s, p_s
	print 'T, p:	', t, p_t
	print '\n',
	
	print 'Tan redshift...'
	high_num	=	np.where( possible_factor___Tanredshift / y_to_fit > 1 )[0].size
	print 'Number of points with M greater and lesser than one:	{0:d} and {1:d}'.format( high_num, common_num - high_num ), '\n'
	r, p_r	=	R( against_what[2], possible_factor___Tanredshift )
	s, p_s	=	S( against_what[2], possible_factor___Tanredshift )
	t, p_t	=	T( against_what[2], possible_factor___Tanredshift )
	print 'R, p:	', r, p_r
	print 'S, p:	', s, p_s
	print 'T, p:	', t, p_t
	print '\n',
	
	print 'My best-fit...'
	high_num	=	np.where( possible_factor___mybestfit / y_to_fit > 1 )[0].size
	print 'Number of points with M greater and lesser than one:	{0:d} and {1:d}'.format( high_num, common_num - high_num ), '\n'
	r, p_r	=	R( against_what[3], possible_factor___mybestfit )
	s, p_s	=	S( against_what[3], possible_factor___mybestfit )
	t, p_t	=	T( against_what[3], possible_factor___mybestfit )
	print 'R, p:	', r, p_r
	print 'S, p:	', s, p_s
	print 'T, p:	', t, p_t
	print '\n',
	
	print 'My redshift...'
	high_num	=	np.where( possible_factor___myredshift / y_to_fit > 1 )[0].size
	print 'Number of points with M greater and lesser than one:	{0:d} and {1:d}'.format( high_num, common_num - high_num ), '\n'
	r, p_r	=	R( against_what[4], possible_factor___myredshift )
	s, p_s	=	S( against_what[4], possible_factor___myredshift )
	t, p_t	=	T( against_what[4], possible_factor___myredshift )
	print 'R, p:	', r, p_r
	print 'S, p:	', s, p_s
	print 'T, p:	', t, p_t
	print '\n',

print '\n\n\n\n'

##	To check for correlations against Ep(1+z) [*not* expected, obviously: this is a consistency check].
print '-------------------------------------------------------------------------------'
print 'Ep(1+z)', '\n\n'

against	=	[]
for i in range(5): against.append(x_to_fit_in_keV)
against	=	np.array(against)
check_for_correlations(against)

ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ E_p \, (1+z) \; \rm{ [keV] } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.scatter( x_to_fit_in_keV, possible_factor___Yonetoku   , s = marker_size  , color = 'b'         , label = r'$ \rm{ Yonetoku                       } $' )
ax.scatter( x_to_fit_in_keV, possible_factor___Tanbestfit , s = marker_size  , color = 'r'         , label = r'$ \rm{ Tan \, (2013) : best \, fit    } $' )
ax.scatter( x_to_fit_in_keV, possible_factor___Tanredshift, s = marker_size  , color = 'y'         , label = r'$ \rm{ Tan \, (2013) : new \, method  } $' )
ax.scatter( x_to_fit_in_keV, possible_factor___mybestfit  , s = marker_size  , color = 'g'         , label = r'$ \rm{ present \, work : best \, fit  } $' )
ax.scatter( x_to_fit_in_keV, possible_factor___myredshift , s = marker_size+2, color = 'darkorange', label = r'$ \rm{ present \, work : redshift     } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_Ep(1+z).png' )
plt.clf()
plt.close()

print '-------------------------------------------------------------------------------\n\n'





##	To check for correlations against measured redshift.
print '-------------------------------------------------------------------------------'
print 'measured redshift', '\n\n'

against	=	[]
for i in range(5): against.append(common_redshift)
against	=	np.array(against)
check_for_correlations(against)

ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.scatter( common_redshift, possible_factor___Yonetoku   , s = marker_size  , color = 'b'         , label = r'$ \rm{ Yonetoku                      } $' )
ax.scatter( common_redshift, possible_factor___Tanbestfit , s = marker_size  , color = 'r'         , label = r'$ \rm{ Tan \, (2013) : best \, fit   } $' )
ax.scatter( common_redshift, possible_factor___Tanredshift, s = marker_size  , color = 'y'         , label = r'$ \rm{ Tan \, (2013) : new \, method } $' )
ax.scatter( common_redshift, possible_factor___mybestfit  , s = marker_size  , color = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
ax.scatter( common_redshift, possible_factor___myredshift , s = marker_size+2, color = 'darkorange', label = r'$ \rm{ present \, work : redshift    } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_measured_redshift.png' )
plt.clf()
plt.close()

ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.errorbar( common_redshift, possible_factor___mybestfit , yerr = possible_factor___mybestfit_error , fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'g'         , markeredgecolor = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
ax.errorbar( common_redshift, possible_factor___myredshift, yerr = possible_factor___myredshift_error, fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'darkorange', markeredgecolor = 'darkorange', label = r'$ \rm{ present \, work : redshift    } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_measured_redshift--my_parameters.png' )
plt.clf()
plt.close()

x_to_fit_log	=	np.log10( (1+common_redshift) )
y_to_fit_log	=	np.log10( possible_factor___mybestfit  )
popt, pcov		=	curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
exponent		=	popt[0]		;	coefficient			=	10**popt[1]
exponent_error	=	pcov[0,0]	;	coefficient_error	=	( 10**pcov[1,1] - 1 ) * coefficient
print '\n\n'
print 'Coefficient		:	',	round(coefficient, 3), round(coefficient_error, 5)
print 'Exponent		:	',		round(   exponent, 3), round(   exponent_error, 3)
ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.errorbar( common_redshift, possible_factor___mybestfit , yerr = possible_factor___mybestfit_error , fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_measured_redshift--my_bestfit.png' )
plt.clf()
plt.close()

r, p_r	=	R( common_redshift, A___mybestfit   *(x_to_fit_in_MeV**eta_mybestfit  ) )
s, p_s	=	S( common_redshift, A___mybestfit   *(x_to_fit_in_MeV**eta_mybestfit  ) )
t, p_t	=	T( common_redshift, A___mybestfit   *(x_to_fit_in_MeV**eta_mybestfit  ) )
print '\n'
print 'R, p:	', r, p_r
print 'S, p:	', s, p_s
print 'T, p:	', t, p_t
print '\n'
ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted } $', fontsize = size_font )
ax.errorbar( common_redshift, A___mybestfit   *(x_to_fit_in_MeV**eta_mybestfit  ), fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'g'         , markeredgecolor = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
ax.errorbar( common_redshift, A___myredshift  *(x_to_fit_in_MeV**eta_myredshift ), fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'darkorange', markeredgecolor = 'darkorange', label = r'$ \rm{ present \, work : redshift    } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_measured_redshift--my_parameters--predicted.png' )
plt.clf()
plt.close()


print '-------------------------------------------------------------------------------\n\n'





##	To check for correlations against pseudo redshift.
print '-------------------------------------------------------------------------------'
print 'pseudo redshift', '\n\n'

k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
term_Fermi	=	k_table['term_Fermi'].data
z_bin		=	np.mean( np.diff(z_sim) )

numerator__delta_pseudo_GRB__first_term		=	common_Fermi_flux_error / common_Fermi_flux								#	defined for each GRB
numerator__delta_pseudo_GRB__second_term	=	common_Epeak_in_MeV_error / common_Epeak_in_MeV							#	defined for each GRB
denominator__delta_pseudo_zsim__first_term	=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  ) + term_Fermi	#	defined for each simulated redshift

numerator__F								=	k_Fermi  *  4*P  *  ( dL_sim**2 )
def discrepancy_and_redchisqrd( A, eta ):
	
	numerator__delta_pseudo_z		=	numerator__delta_pseudo_GRB__first_term  +  eta * numerator__delta_pseudo_GRB__second_term		#	defined for each GRB
	denominator__delta_pseudo_z		=	denominator__delta_pseudo_zsim__first_term  +  (  (eta+2) / (1+z_sim)  )							#	defined for each simulated redshift
	
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
	
	discrepancy	=	np.sum( ( common_redshift - pseudo_redshifts )**2 )
	chisquared, dof, redchisqrd	=	mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
	
	
	return pseudo_redshifts, pseudo_redshifts_error, discrepancy, chisquared, dof, redchisqrd


print '\n\n\n\n'
against	=	[]
for i in range(5):
	if i == 0:
		returned	=	discrepancy_and_redchisqrd( A___Yonetoku, eta_Yonetoku )		;	pseudo_redshifts	=	returned[0]	;	pseudo_redshifts_error	=	returned[1]
		against.append(  pseudo_redshifts  )
		pseudo_redshifts___Yonetoku				=	pseudo_redshifts
		pseudo_redshifts___Yonetoku_error		=	pseudo_redshifts_error
		print i, returned[2], mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
		
	if i == 1:
		returned	=	discrepancy_and_redchisqrd( A___Tanbestfit, eta_Tanbestfit )	;	pseudo_redshifts	=	returned[0]	;	pseudo_redshifts_error	=	returned[1]
		against.append(  pseudo_redshifts  )
		pseudo_redshifts___Tanbestfit			=	pseudo_redshifts
		pseudo_redshifts___Tanbestfit_error		=	pseudo_redshifts_error
		print i, returned[2], mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
		
	if i == 2:
		returned	=	discrepancy_and_redchisqrd( A___Tanredshift, eta_Tanredshift )	;	pseudo_redshifts	=	returned[0]	;	pseudo_redshifts_error	=	returned[1]
		against.append(  pseudo_redshifts  )
		pseudo_redshifts___Tanredshift			=	pseudo_redshifts
		pseudo_redshifts___Tanredshift_error	=	pseudo_redshifts_error
		print i, returned[2], mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
		
	if i == 3:
		returned	=	discrepancy_and_redchisqrd( A___mybestfit, eta_mybestfit )		;	pseudo_redshifts	=	returned[0]	;	pseudo_redshifts_error	=	returned[1]
		against.append(  pseudo_redshifts  )
		pseudo_redshifts___mybestfit			=	pseudo_redshifts
		pseudo_redshifts___mybestfit_error		=	pseudo_redshifts_error
		print i, returned[2], mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
		
	if i == 4:
		returned	=	discrepancy_and_redchisqrd( A___myredshift, eta_myredshift )	;	pseudo_redshifts	=	returned[0]	;	pseudo_redshifts_error	=	returned[1]
		against.append(  pseudo_redshifts  )
		pseudo_redshifts___myredshift			=	pseudo_redshifts
		pseudo_redshifts___myredshift_error		=	pseudo_redshifts_error
		print i, returned[2], mf.reduced_chisquared( common_redshift, pseudo_redshifts, pseudo_redshifts_error, 2 )
print '\n\n\n\n'

ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ pseudo \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.scatter( pseudo_redshifts___Yonetoku   , possible_factor___Yonetoku   , s = marker_size  , color = 'b'         , label = r'$ \rm{ Yonetoku                      } $' )
ax.scatter( pseudo_redshifts___Tanbestfit , possible_factor___Tanbestfit , s = marker_size  , color = 'r'         , label = r'$ \rm{ Tan \, (2013) : best \, fit   } $' )
ax.scatter( pseudo_redshifts___Tanredshift, possible_factor___Tanredshift, s = marker_size  , color = 'y'         , label = r'$ \rm{ Tan \, (2013) : new \, method } $' )
ax.scatter( pseudo_redshifts___mybestfit  , possible_factor___mybestfit  , s = marker_size  , color = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
ax.scatter( pseudo_redshifts___myredshift , possible_factor___myredshift , s = marker_size+2, color = 'darkorange', label = r'$ \rm{ present \, work : redshift    } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_pseudo_redshift.png' )
plt.clf()
plt.close()
against	=	np.array(against)
check_for_correlations(against)


ax	=	plt.subplot( 111 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ pseudo \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ \rm{ predicted / observed } $', fontsize = size_font )
ax.errorbar( pseudo_redshifts___mybestfit , possible_factor___mybestfit , xerr = pseudo_redshifts___mybestfit_error , yerr = possible_factor___mybestfit_error , fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'g'         , markeredgecolor = 'g'         , label = r'$ \rm{ present \, work : best \, fit } $' )
ax.errorbar( pseudo_redshifts___myredshift, possible_factor___myredshift, xerr = pseudo_redshifts___myredshift_error, yerr = possible_factor___myredshift_error, fmt = '.', ms = marker_size, color =  'silver', markerfacecolor = 'darkorange', markeredgecolor = 'darkorange', label = r'$ \rm{ present \, work : redshift    } $' )
ax.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/looking_for_possible_systematics/scatter_with_pseudo_redshift--my_parameters.png' )
plt.clf()
plt.close()

print '-------------------------------------------------------------------------------\n\n'




print '\n\n-------------------------------------------------------------------------------'
print 'The trend in the pseudo redshifts...', '\n\n'

def fit_the_redshift_trend( pseudo_redshifts ):
	
	r, p_r	=	R( common_redshift, pseudo_redshifts )
	s, p_s	=	S( common_redshift, pseudo_redshifts )
	t, p_t	=	T( common_redshift, pseudo_redshifts )
	print 'R, p			:	', r, p_r
	print 'S, p			:	', s, p_s
	print 'T, p			:	', t, p_t
	
	x_to_fit_log	=	np.log10( pseudo_redshifts )
	y_to_fit_log	=	np.log10( common_redshift  )
	
	#	plt.scatter( x_to_fit_log, y_to_fit_log )
	#	plt.show()
	
	popt, pcov		=	curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
	exponent		=	popt[0]		;	coefficient			=	10**popt[1]
	exponent_error	=	pcov[0,0]	;	coefficient_error	=	( 10**pcov[1,1] - 1 ) * coefficient
	print 'Coefficient		:	',	round(coefficient, 3), round(coefficient_error, 5)
	print 'Exponent		:	',		round(   exponent, 3), round(   exponent_error, 3)
	
	return coefficient, coefficient_error, exponent, exponent_error
	

def plot_redshift_vs_redshift( pseudo_redshifts, pseudo_redshifts_error, coefficient, coefficient_error, exponent, exponent_error ):
	
	reconciled			=	coefficient*(pseudo_redshifts**exponent)
	reconciled_error	=	exponent*(pseudo_redshifts_error/pseudo_redshifts)  +  coefficient_error/coefficient  +  np.log(pseudo_redshifts)*exponent_error
	reconciled_error	=	reconciled_error * reconciled
	ratio				=	reconciled / common_redshift
	ratio_error			=	( reconciled_error / reconciled ) * ratio
	
	print 'Ratio min and max	:	', round( ratio.min(), 3 ), round( ratio.max(), 3 )
	print 'Reduced chi-squared	:	', mf.reduced_chisquared( common_redshift, reconciled, reconciled_error, 4 )[2]
	
	
	z_min		=	1e-1
	z_max		=	2e1
	
	ax	=	plt.subplot(111)
	ax.set_xscale( 'log' )
	ax.set_yscale( 'log' )
	ax.set_xlim( z_min, z_max )
	ax.set_ylim( z_min, z_max )
	ax.set_ylabel( r'$ \rm{ reconciled / measured } $' , fontsize = size_font )
	ax.set_xlabel( r'$ \rm{ pseudo \; \, redshift } $' , fontsize = size_font )
	ax.errorbar( pseudo_redshifts, ratio, yerr = ratio_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
	ax.axhline( y = 1, linestyle = 'dashed', color = 'k' )



print 'Yonetoku...'
returned	=	fit_the_redshift_trend( pseudo_redshifts___Yonetoku )
coefficient			=	returned[0]		;	exponent		=	returned[2]
coefficient_error	=	returned[1]		;	exponent_error	=	returned[3]
plot_redshift_vs_redshift( pseudo_redshifts___Yonetoku, pseudo_redshifts___Yonetoku_error    , coefficient, coefficient_error, exponent, exponent_error )
plt.savefig( './../plots/looking_for_possible_systematics/reconciling_the_pseudo_redshifts/Yonetoku--reconciled.png' )
plt.clf()
plt.close()
print '\n',

print 'Tan best-fit...'
returned	=	fit_the_redshift_trend( pseudo_redshifts___Tanbestfit )
coefficient			=	returned[0]		;	exponent		=	returned[2]
coefficient_error	=	returned[1]		;	exponent_error	=	returned[3]
plot_redshift_vs_redshift( pseudo_redshifts___Tanbestfit, pseudo_redshifts___Tanbestfit_error , coefficient, coefficient_error, exponent, exponent_error )
plt.savefig( './../plots/looking_for_possible_systematics/reconciling_the_pseudo_redshifts/Tanbestfit--reconciled.png' )
plt.clf()
plt.close()
print '\n',

print 'Tan redshift...'
returned	=	fit_the_redshift_trend( pseudo_redshifts___Tanredshift )
coefficient			=	returned[0]		;	exponent		=	returned[2]
coefficient_error	=	returned[1]		;	exponent_error	=	returned[3]
plot_redshift_vs_redshift( pseudo_redshifts___Tanredshift, pseudo_redshifts___Tanredshift_error, coefficient, coefficient_error, exponent, exponent_error )
plt.savefig( './../plots/looking_for_possible_systematics/reconciling_the_pseudo_redshifts/Tanredshift--reconciled.png' )
plt.clf()
plt.close()
print '\n',

print 'My best-fit...'
returned	=	fit_the_redshift_trend( pseudo_redshifts___mybestfit )
coefficient			=	returned[0]		;	exponent		=	returned[2]
coefficient_error	=	returned[1]		;	exponent_error	=	returned[3]
plot_redshift_vs_redshift( pseudo_redshifts___mybestfit, pseudo_redshifts___mybestfit_error   , coefficient, coefficient_error, exponent, exponent_error )
plt.savefig( './../plots/looking_for_possible_systematics/reconciling_the_pseudo_redshifts/mybestfit--reconciled.png' )
plt.clf()
plt.close()
print '\n',

print 'My redshift...'
returned	=	fit_the_redshift_trend( pseudo_redshifts___myredshift )
coefficient			=	returned[0]		;	exponent		=	returned[2]
coefficient_error	=	returned[1]		;	exponent_error	=	returned[3]
plot_redshift_vs_redshift( pseudo_redshifts___myredshift, pseudo_redshifts___myredshift_error, coefficient, coefficient_error, exponent, exponent_error )
plt.savefig( './../plots/looking_for_possible_systematics/reconciling_the_pseudo_redshifts/myredshift--reconciled.png' )
plt.clf()
plt.close()
print '\n',

print '-------------------------------------------------------------------------------\n\n'
