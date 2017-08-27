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
A___mybestfit	=	4.780		#	my best-fit
eta_mybestfit	=	1.229		#	my best-fit

#	A	=	A___Yonetoku
#	eta	=	eta_Yonetoku
#	A	=	A___Tanbestfit
#	eta	=	eta_Tanbestfit
#	A	=	A___Tanredshift
#	eta	=	eta_Tanredshift
A	=	A___mybestfit
eta	=	eta_mybestfit


logL_bin	=	0.5
logL_min	=	-5
logL_max	=	5


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	07	# The size of markers in scatter plots.
al			=	0.8	# The brightness of plots.
z_min		=	1e-1 #	for the purposes of plotting
z_max		=	1e+1 #	for the purposes of plotting
x_in_keV_min	=	1e00	;	x_in_keV_max	=	2e04	#	Ep(1+z), min & max.
y_in_eps_min	=	1e49	;	y_in_eps_max	=	1e56	#	L_iso  , min & max.


####################################################################################################################################################






####################################################################################################################################################
########	Defining the functions.


def choose( bigger, smaller ):
	
	
	indices = []
	
	for i, s in enumerate( smaller ):
		ind	=	np.where(bigger == s)[0][0]		# the index is of the bigger array.
		indices.append( ind )
	
	
	return np.array(indices)



k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim		=	k_table['z'].data
dL_sim		=	k_table['dL'].data
k_Fermi		=	k_table['k_Fermi'].data
k_Swift		=	k_table['k_Swift'].data
term_Fermi	=	k_table['term_Fermi'].data
term_Swift	=	k_table['term_Swift'].data
z_bin		=	np.mean( np.diff(z_sim) )

#~ ax	=	plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ \rm{ k } $', fontsize = size_font+2, rotation = 0, labelpad = padding )
#~ ax.plot( z_sim, k_Fermi, color = 'r', ms = marker_size, label = r'$ Fermi $' )
#~ ax.plot( z_sim, k_Swift, color = 'b', ms = marker_size, label = r'$ Swift $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/pseudo_calculations/k_correction--both.png' )
#~ plt.clf()
#~ plt.close()

numerator_F		=	4*P  *  ( dL_sim**2 )
denominator_F	=	(1+z_sim)**eta

denominator__delta_pseudo_z__Fermi		=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )  +  (  (eta+2) / (1+z_sim)  ) + term_Fermi
numerator_F__Fermi	=	k_Fermi  *  numerator_F
F_Fermi		=	numerator_F__Fermi / denominator_F
F_Fermi		=	F_Fermi * (cm_per_Mpc**2) / L_norm

denominator__delta_pseudo_z__Swift		=	(2/dL_sim) * (C/H_0) / np.sqrt(  CC + (1-CC)*( (1+z_sim)**3 )  )  +  (  (eta+2) / (1+z_sim)  ) + term_Swift
numerator_F__Swift	=	k_Swift  *  numerator_F
F_Swift		=	numerator_F__Swift / denominator_F
F_Swift		=	F_Swift * (cm_per_Mpc**2) / L_norm
F_Swift		=	F_Swift * erg_per_keV

#~ ax	=	plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ \rm{ F } $', fontsize = size_font+2, rotation = 0, labelpad = padding )
#~ ax.plot( z_sim, F_Fermi, color = 'r', ms = marker_size, label = r'$ Fermi $' )
#~ ax.plot( z_sim, F_Swift, color = 'b', ms = marker_size, label = r'$ Swift $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/pseudo_calculations/F(z)--both.png' )
#~ plt.clf()
#~ plt.close()



def estimate_pseudo_redshift_and_Luminosity__Fermi( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, alpha, beta ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
		
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Fermi - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__Fermi[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', pseudo_redshifts.min(), pseudo_redshifts.max(), '\n'
	
	#~ ax = plt.subplot(111)
	#~ ax.set_xscale('log')
	#~ ax.set_yscale('log')
	#~ ax.set_title( r'$ Fermi $' )
	#~ ax.plot( z_sim, F_Fermi, color = 'k', label = r'$ F $' )
	#~ ax.scatter( pseudo_redshifts, RHSs, color = 'r', label = r'$ RHS $' )
	#~ plt.legend()
	#~ plt.show()
	
	
	Epeak_into_oneplusz	=	Epeak_in_MeV*(1+pseudo_redshifts)
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		al		=	alpha[j]
		be		=	beta[j]
		Ep_keV	=	Epeak_in_MeV[j] * 1e3
		Luminosity[j]	=	sf.Liso_with_known_spectral_parameters__Fermi( ep, al, be, Ep_keV, z )
	Luminosity			=	Luminosity * ( cm_per_Mpc**2 )
	#	Luminosity_error	=	Luminosity * (flux_error/flux)
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshift)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	inds_to_delete	=	np.where( pseudo_redshifts >= 10 )[0]
	print '# of GRBs with redshift > 10	:	', inds_to_delete.size
	#~ print 'Corresponding indices		:	', inds_to_delete, '\n'
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', percentage_error.min(), percentage_error.max()
	print 'Percentage error, mean		:	', percentage_error.mean()
	print 'Luminosity error, mean:			', 100 * (Luminosity_error/Luminosity).mean()
	inds_huge_errors	=	np.where( percentage_error >= percentage_cutoff )[0]
	print '# of GRBs with errors > {0:d}%	:	{1:d}'.format(percentage_cutoff, inds_huge_errors.size)
	#~ print 'Corresponding indices		:	', inds_huge_errors
	#~ print 'Index with largest error	:	', np.argmax(percentage_error), '\n'
	#~ plt.loglog( percentage_error )
	#~ plt.show()
	
	#~ inds_to_delete	=	np.append( inds_to_delete, inds_huge_errors )
	#~ pseudo_redshifts			=	np.delete( pseudo_redshifts, inds_to_delete )
	#~ pseudo_redshifts_error	=	np.delete( pseudo_redshifts_error, inds_to_delete )
	#~ flux						=	np.delete( flux, inds_to_delete )
	#~ flux_error					=	np.delete( flux_error, inds_to_delete )
	#~ Epeak_in_MeV				=	np.delete( Epeak_in_MeV, inds_to_delete )
	#~ Epeak_in_MeV_error			=	np.delete( Epeak_in_MeV_error, inds_to_delete )
	#~ Luminosity					=	np.delete( Luminosity, inds_to_delete )
	#~ Luminosity_error			=	np.delete( Luminosity_error, inds_to_delete )
	
	#~ percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	#~ print 'After deleting,'
	#~ print 'Percentage error, min and max	:	', percentage_error.min()   ,    percentage_error.max(), '\n'
	#~ print 'pseudo redshift , min and max	:	', pseudo_redshifts.min(), pseudo_redshifts.max(), '\n'
	#~ plt.loglog( percentage_error )
	#~ plt.show()
	
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error



def estimate_pseudo_redshift_and_Luminosity__Swift( name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error ):
	
	
	numerator__delta_pseudo_z		=	( flux_error / flux )  +  eta * ( Epeak_in_MeV_error / Epeak_in_MeV )		#	defined for each GRB	
	RHSs	=	( A / flux ) * ( Epeak_in_MeV**eta )
	
	pseudo_redshifts		=	np.zeros( RHSs.size )
	pseudo_redshifts_error	=	np.zeros( RHSs.size )
	for j, RHS in enumerate(RHSs):
		
		array	=	np.abs( F_Swift - RHS )
		ind		=	np.where(  array == array.min()  )[0][0]
		pseudo_redshift			=	z_sim[ind]
		pseudo_redshift_error	=	numerator__delta_pseudo_z[j] / denominator__delta_pseudo_z__Swift[ind]
		
		pseudo_redshifts[j]			=	pseudo_redshift
		pseudo_redshifts_error[j]	=	pseudo_redshift_error
	
	print 'pseudo redshift , min and max	:	', pseudo_redshifts.min(), pseudo_redshifts.max(), '\n'
		
	#~ ax = plt.subplot(111)
	#~ ax.set_xscale('log')
	#~ ax.set_yscale('log')
	#~ ax.set_title( r'$ Swift $' )
	#~ ax.plot( z_sim, F_Swift, color = 'k', label = r'$ F $' )
	#~ ax.scatter( pseudo_redshifts, RHSs, color = 'r', label = r'$ RHS $' )
	#~ plt.legend()
	#~ plt.show()
	
	
	Luminosity	=	flux.copy()
	for j, z in enumerate(pseudo_redshifts):
		ep		=	flux[j]
		Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( ep, z )
	Luminosity			=	Luminosity * (cm_per_Mpc**2) * erg_per_keV
	#	Luminosity_error	=	Luminosity * (flux_error/flux)
	Luminosity_error	=	Luminosity * ( eta/(Epeak_in_MeV*(1+pseudo_redshifts)) )  *  ( (1+pseudo_redshifts)*Epeak_in_MeV_error + Epeak_in_MeV*pseudo_redshifts_error )
	
	inds_to_delete	=	np.where( pseudo_redshifts >= 10 )[0]
	print '# of GRBs with redshift > 10	:	', inds_to_delete.size
	#~ print 'Corresponding indices		:	', inds_to_delete, '\n'
	
	percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	percentage_cutoff	=	100
	print 'Percentage error, min and max	:	', percentage_error.min(), percentage_error.max()
	print 'Percentage error, mean		:	', percentage_error.mean()
	print 'Luminosity error, mean:			', 100 * (Luminosity_error/Luminosity).mean()
	inds_huge_errors	=	np.where( percentage_error >= percentage_cutoff )[0]
	print '# of GRBs with errors > {0:d}%	:	{1:d}'.format(percentage_cutoff, inds_huge_errors.size)
	#~ print 'Corresponding indices		:	', inds_huge_errors
	#~ print 'Index with largest error	:	', np.argmax(percentage_error), '\n'
	#~ plt.loglog( percentage_error )
	#~ plt.show()
	#~ 
	#~ inds_to_delete	=	np.append( inds_to_delete, inds_huge_errors )
	#~ pseudo_redshifts			=	np.delete( pseudo_redshifts, inds_to_delete )
	#~ pseudo_redshifts_error	=	np.delete( pseudo_redshifts_error, inds_to_delete )
	#~ flux						=	np.delete( flux, inds_to_delete )
	#~ flux_error					=	np.delete( flux_error, inds_to_delete )
	#~ Epeak_in_MeV				=	np.delete( Epeak_in_MeV, inds_to_delete )
	#~ Epeak_in_MeV_error			=	np.delete( Epeak_in_MeV_error, inds_to_delete )
	#~ Luminosity					=	np.delete( Luminosity, inds_to_delete )
	#~ Luminosity_error			=	np.delete( Luminosity_error, inds_to_delete )
	#~ 
	#~ percentage_error	=	100 * ( pseudo_redshifts_error / pseudo_redshifts )
	#~ print 'After deleting,'
	#~ print 'Percentage error, min and max	:	', percentage_error.min()   ,    percentage_error.max(), '\n'
	#~ print 'pseudo redshift , min and max	:	', pseudo_redshifts.min(), pseudo_redshifts.max(), '\n'
	#~ plt.loglog( percentage_error )
	#~ plt.show()
	
	
	return name, flux, flux_error, Epeak_in_MeV, Epeak_in_MeV_error, pseudo_redshifts, pseudo_redshifts_error, Luminosity, Luminosity_error



####################################################################################################################################################






####################################################################################################################################################
########	Reading the data.


Swift_all_GRBs_table			=	ascii.read( './../tables/Swift_GRBs--all.txt', format = 'fixed_width' )
Swift_all_name					=	Swift_all_GRBs_table['Swift name'].data
Swift_all_T90					=	Swift_all_GRBs_table['BAT T90'].data
Swift_all_flux					=	Swift_all_GRBs_table['BAT Phoflux'].data
Swift_all_flux_error			=	Swift_all_GRBs_table['BAT Phoflux_error'].data
Swift_all_num					=	Swift_all_name.size


Swift_wkr_GRBs_table			=	ascii.read( './../tables/Swift_GRBs--wkr.txt', format = 'fixed_width' )
Swift_wkr_name					=	Swift_wkr_GRBs_table['Swift name'].data
Swift_wkr_redhsift				=	Swift_wkr_GRBs_table['redshift'].data
Swift_wkr_T90					=	Swift_wkr_GRBs_table['BAT T90'].data
Swift_wkr_flux					=	Swift_wkr_GRBs_table['BAT Phoflux'].data
Swift_wkr_flux_error			=	Swift_wkr_GRBs_table['BAT Phoflux_error'].data
Swift_wkr_num					=	Swift_wkr_name.size
Swift_wkr_num					=	Swift_wkr_name.size
#~ print 'Ascending order Swift fluxes:	'
#~ print np.sort(Swift_all_flux)[0:20]
#~ print np.sort(Swift_wkr_flux)[0:10]

#~ print np.where(Swift_all_flux < 0.02)[0]
#~ indices_to_delete	=	np.where(Swift_all_flux < 0.2)[0]
#~ Swift_all_name		=	np.delete( Swift_all_name      , indices_to_delete )
#~ Swift_all_T90		=	np.delete( Swift_all_T90       , indices_to_delete )
#~ Swift_all_flux		=	np.delete( Swift_all_flux      , indices_to_delete )
#~ Swift_all_flux_error=	np.delete( Swift_all_flux_error, indices_to_delete )
#~ Swift_all_num		=	Swift_all_name.size



Fermi_GRBs_table				=	ascii.read( './../tables/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width' )
Fermi_name						=	Fermi_GRBs_table['Fermi name'].data
Fermi_T90						=	Fermi_GRBs_table['GBM T90'].data
Fermi_T90_error					=	Fermi_GRBs_table['GBM T90_error'].data
Fermi_flux						=	Fermi_GRBs_table['GBM flux'].data
Fermi_flux_error				=	Fermi_GRBs_table['GBM flux_error'].data
Fermi_Epeak      				=	Fermi_GRBs_table['Epeak'].data
Fermi_Epeak_error				=	Fermi_GRBs_table['Epeak_error'].data
Fermi_alpha						=	Fermi_GRBs_table['alpha'].data
Fermi_alpha_error				=	Fermi_GRBs_table['alpha_error'].data
Fermi_beta						=	Fermi_GRBs_table['beta'].data
Fermi_beta_error				=	Fermi_GRBs_table['beta_error'].data
Fermi_num						=	Fermi_name.size
#~ print 'Ascending order Fermi fluxes:	'
#~ print np.sort(Fermi_flux)[0:10]


common_all_GRBs_table			=	ascii.read( './../tables/common_GRBs--all.txt', format = 'fixed_width' )
common_all_ID					=	common_all_GRBs_table['common ID'].data
common_all_Swift_name			=	common_all_GRBs_table['Swift name'].data
common_all_Fermi_name			=	common_all_GRBs_table['Fermi name'].data
common_all_Swift_T90			=	common_all_GRBs_table['BAT T90'].data
common_all_Fermi_T90			=	common_all_GRBs_table['GBM T90'].data
common_all_Fermi_T90_error		=	common_all_GRBs_table['GBM T90_error'].data
common_all_Fermi_flux			=	common_all_GRBs_table['GBM flux'].data
common_all_Fermi_flux_error		=	common_all_GRBs_table['GBM flux_error'].data
common_all_Epeak				=	common_all_GRBs_table['Epeak'].data				#	in keV.
common_all_Epeak_error			=	common_all_GRBs_table['Epeak_error'].data		#	in keV.
common_all_alpha				=	common_all_GRBs_table['alpha'].data
common_all_alpha_error			=	common_all_GRBs_table['alpha_error'].data
common_all_beta					=	common_all_GRBs_table['beta'].data
common_all_beta_error			=	common_all_GRBs_table['beta_error'].data
common_all_num					=	common_all_ID.size

common_wkr_GRBs_table			=	ascii.read( './../tables/common_GRBs--wkr.txt', format = 'fixed_width' )
common_wkr_ID					=	common_wkr_GRBs_table['common ID'].data
common_wkr_Swift_name			=	common_wkr_GRBs_table['Swift name'].data
common_wkr_Fermi_name			=	common_wkr_GRBs_table['Fermi name'].data
common_wkr_Swift_T90			=	common_wkr_GRBs_table['BAT T90'].data
common_wkr_Fermi_T90			=	common_wkr_GRBs_table['GBM T90'].data
common_wkr_Fermi_T90_error		=	common_wkr_GRBs_table['GBM T90_error'].data
common_wkr_redshift				=	common_wkr_GRBs_table['redshift'].data
common_wkr_Fermi_flux			=	common_wkr_GRBs_table['GBM flux'].data
common_wkr_Fermi_flux_error		=	common_wkr_GRBs_table['GBM flux_error'].data
common_wkr_Epeak				=	common_wkr_GRBs_table['Epeak'].data				#	in keV.
common_wkr_Epeak_error			=	common_wkr_GRBs_table['Epeak_error'].data		#	in keV.
common_wkr_alpha				=	common_wkr_GRBs_table['alpha'].data
common_wkr_alpha_error			=	common_wkr_GRBs_table['alpha_error'].data
common_wkr_beta					=	common_wkr_GRBs_table['beta'].data
common_wkr_beta_error			=	common_wkr_GRBs_table['beta_error'].data
common_wkr_Luminosity			=	common_wkr_GRBs_table['Luminosity'].data
common_wkr_Luminosity_error		=	common_wkr_GRBs_table['Luminosity_error'].data
common_wkr_num					=	common_wkr_ID.size


####################################################################################################################################################






####################################################################################################################################################
########	For the Fermi GRBs, including those common with Swift (since they have spectra), except those with known redshifts (L already known).





####	First for the long ones.

##	Finding all the long GRBs.
print 'Number of common GRBs				:	', common_all_num

inds_long_in_universal_common_sample_by_applying_Swift_criterion		=	np.where( common_all_Swift_T90 >= T90_cut )[0]	# these indices run over the sample of all common GRBs (i.e. with/without redshift).
print 'Number of long ones amongst them 		:	', inds_long_in_universal_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion			=	common_all_Fermi_name[inds_long_in_universal_common_sample_by_applying_Swift_criterion]
inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion		=	choose( Fermi_name, Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.

#	print '\n\n'
#	print inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion.size
#	print inds_long_in_universal_common_sample_by_applying_Swift_criterion
#	print inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion
#	print inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion - inds_long_in_universal_common_sample_by_applying_Swift_criterion
#	print ( inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion - inds_long_in_universal_common_sample_by_applying_Swift_criterion >= 0 ).all()


print 'Total number of GRBs in the Fermi sample	:	', Fermi_num
inds_in_Fermi_common_all		=	choose( Fermi_name, common_all_Fermi_name )	# these indices run over the universal Fermi sample.
print 'Number of common GRBs				:	', inds_in_Fermi_common_all.size
inds_in_Fermi_uncommon_all		=	np.delete( np.arange(Fermi_num), inds_in_Fermi_common_all )	# these indices run over the universal Fermi sample.
print 'Out of which those detected only by Fermi	:	', inds_in_Fermi_uncommon_all.size

#	ascii.write(  Table( [ Fermi_name[inds_in_Fermi_uncommon_all] ], names = ['Fermi name'] ),  './tables/Fermi_only_names.txt', format = 'fixed_width', overwrite = True  )

Fermi_T90_uncommon_all	=	Fermi_T90[ inds_in_Fermi_uncommon_all]
Fermi_name_uncommon_all	=	Fermi_name[inds_in_Fermi_uncommon_all]

inds_long_in_Fermi_only_sample				=	np.where( Fermi_T90_uncommon_all >= T90_cut )[0]	# these indices run over the Fermi-only sample (with/without redshift).
Fermi_name_for_long_in_Fermi_only_sample	=	Fermi_name_uncommon_all[inds_long_in_Fermi_only_sample]
print 'Long in Fermi-only sample, by Fermi criterion	:	', inds_long_in_Fermi_only_sample.size

inds_long_in_Fermi_full_sample	=	choose( Fermi_name, Fermi_name_for_long_in_Fermi_only_sample )	# these indices run over the universal Fermi sample.

#	print '\n\n'
#	print inds_long_in_Fermi_full_sample.size
#	print inds_long_in_Fermi_only_sample
#	print inds_long_in_Fermi_full_sample
#	print inds_long_in_Fermi_full_sample - inds_long_in_Fermi_only_sample
#	print ( inds_long_in_Fermi_full_sample - inds_long_in_Fermi_only_sample >= 0 ).all()

#	print 'Check: any common amongst these disjoint sets?	:	', np.intersect1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion ).size
#	print 'Check: just appending the two arrays		:	', np.append( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion ).size

inds_long_in_Fermi	=	np.union1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_long_in_universal_common_sample_by_applying_Swift_criterion )		# these indices run over the universal Fermi sample.
print 'Total number of long GRBs in the Fermi sample	:	', inds_long_in_Fermi.size



##	Finding only the long GRBs without redshift.
print '\n\n'
print 'Number of common GRBs with known redshift	:	', common_wkr_num

inds_long_in_redshift_common_sample_by_applying_Swift_criterion		=	np.where( common_wkr_Swift_T90 >= T90_cut )[0]	# these indices run over the sample of common GRBs with known redshift.
print 'Number of long ones amongst them 		:	', inds_long_in_redshift_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_long_in_redshift_common_sample_by_applying_Swift_criterion			=	common_wkr_Fermi_name[inds_long_in_redshift_common_sample_by_applying_Swift_criterion]
inds_amongst_common_all_for_those_wkr	=	choose( Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion, Fermi_name_for_long_in_redshift_common_sample_by_applying_Swift_criterion ) 
#	print inds_amongst_common_all_for_those_wkr.size

inds_without_redshift_amongst_common_sample		=	np.delete( np.arange(inds_long_in_universal_common_sample_by_applying_Swift_criterion.size), inds_amongst_common_all_for_those_wkr )	# these indices run over all the common GRBs that are long (only, by Swift criterion).
Fermi_name_for_common_long_GRBs_without_redshift	=	Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion[inds_without_redshift_amongst_common_sample]
print 'Common GRBs with unknown redshift		:	', Fermi_name_for_common_long_GRBs_without_redshift.size

#	inds_long_amongst_common_wkr	=	choose( common_wkr_Fermi_name, Fermi_name_for_long_in_redshift_common_sample_by_applying_Swift_criterion )
#	inds_short_amongst_common_wkr	=	np.delete( np.arange(common_wkr_num), inds_long_amongst_common_wkr )
#	print inds_short_amongst_common_wkr, common_wkr_Fermi_name[inds_short_amongst_common_wkr]
#	print ( common_wkr_Fermi_name[inds_long_amongst_common_wkr] == Fermi_name_for_long_in_universal_common_sample_by_applying_Swift_criterion[inds_amongst_common_all_for_those_wkr] ).all()

inds_in_Fermi_for_common_long_GRBs_without_redshift	=	choose( Fermi_name, Fermi_name_for_common_long_GRBs_without_redshift )	# these indices run over the universal Fermi sample.

#	print 'Check: any common amongst these disjoint sets?	:	', np.intersect1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_common_long_GRBs_without_redshift ).size
#	print 'Check: just appending the two arrays		:	', np.append( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_common_long_GRBs_without_redshift ).size

inds_long_in_Fermi_without_redshift	=	np.union1d( inds_long_in_Fermi_full_sample, inds_in_Fermi_for_common_long_GRBs_without_redshift )		# these indices run over the universal Fermi sample.
print 'Total number of Fermi l-GRBs without redshift	:	', inds_long_in_Fermi_without_redshift.size
print '\n\n'

print '\n\n\n\n'









####	Similarly for short GRBs in Fermi.
print 'Number of common GRBs				:	', common_all_num

inds_short_in_universal_common_sample_by_applying_Swift_criterion		=	np.where( common_all_Swift_T90 < T90_cut )[0]	# these indices run over the sample of all common GRBs (i.e. with/without redshift).
print 'Number of short ones amongst them 		:	', inds_short_in_universal_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion			=	common_all_Fermi_name[inds_short_in_universal_common_sample_by_applying_Swift_criterion]
inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion	=	choose( Fermi_name, Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.

print 'Total number of GRBs in the Fermi sample	:	', Fermi_num
print 'Out of which those detected only by Fermi	:	', inds_in_Fermi_uncommon_all.size


inds_short_in_Fermi_only_sample				=	np.where( Fermi_T90_uncommon_all < T90_cut )[0]	# these indices run over the Fermi-only sample (with/without redshift).
Fermi_name_for_short_in_Fermi_only_sample	=	Fermi_name_uncommon_all[inds_short_in_Fermi_only_sample]
print 'Short in Fermi-only sample, by Fermi criterion	:	', inds_short_in_Fermi_only_sample.size

inds_short_in_Fermi_full_sample	=	choose( Fermi_name, Fermi_name_for_short_in_Fermi_only_sample )	# these indices run over the universal Fermi sample.

#	print 'Check: any common amongst these disjoint sets?	:	', np.intersect1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion ).size
#	print 'Check: just appending the two arrays		:	', np.append( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion ).size

inds_short_in_Fermi				=	np.union1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_short_in_universal_common_sample_by_applying_Swift_criterion )	# these indices run over the universal Fermi sample.
print 'Total number of short GRBs in the Fermi sample	:	', inds_short_in_Fermi.size



##	Finding only the short GRBs without redshift.
print '\n\n'
print 'Number of common GRBs with known redshift	:	', common_wkr_num

inds_short_in_redshift_common_sample_by_applying_Swift_criterion		=	np.where( common_wkr_Swift_T90 < T90_cut )[0]	# these indices run over the sample of common GRBs with known redshift.
print 'Number of short ones amongst them 		:	', inds_short_in_redshift_common_sample_by_applying_Swift_criterion.size

Fermi_name_for_short_in_redshift_common_sample_by_applying_Swift_criterion		=	common_wkr_Fermi_name[inds_short_in_redshift_common_sample_by_applying_Swift_criterion]
inds_amongst_common_all_for_those_wkr	=	choose( Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion, Fermi_name_for_short_in_redshift_common_sample_by_applying_Swift_criterion ) 

inds_without_redshift_amongst_common_sample		=	np.delete( np.arange(inds_short_in_universal_common_sample_by_applying_Swift_criterion.size), inds_amongst_common_all_for_those_wkr )	# these indices run over all the common GRBs that are long (only, by Swift criterion).
Fermi_name_for_common_short_GRBs_without_redshift	=	Fermi_name_for_short_in_universal_common_sample_by_applying_Swift_criterion[inds_without_redshift_amongst_common_sample]
print 'Common GRBs with unknown redshift		:	', Fermi_name_for_common_short_GRBs_without_redshift.size

inds_in_Fermi_for_common_short_GRBs_without_redshift	=	choose( Fermi_name, Fermi_name_for_common_short_GRBs_without_redshift )	# these indices run over the universal Fermi sample..

#	print 'Check: any common amongst these disjoint sets?	:	', np.intersect1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_common_short_GRBs_without_redshift ).size
#	print 'Check: just appending the two arrays		:	', np.append( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_common_short_GRBs_without_redshift ).size

inds_short_in_Fermi_without_redshift	=	np.union1d( inds_short_in_Fermi_full_sample, inds_in_Fermi_for_common_short_GRBs_without_redshift )		# these indices run over the universal Fermi sample.
print 'Total number of Fermi s-GRBs without redshift	:	', inds_short_in_Fermi_without_redshift.size
print '\n\n'

print '\n\n\n\n'








#	print 'Check: any common amongst these disjoint sets?	:	', np.intersect1d( inds_long_in_Fermi_without_redshift, inds_short_in_Fermi_without_redshift ).size, '\n\n'

Fermi_long_name					=	Fermi_name[inds_long_in_Fermi_without_redshift]
Fermi_long_flux					=	Fermi_flux[inds_long_in_Fermi_without_redshift]
Fermi_long_flux_error			=	Fermi_flux_error[inds_long_in_Fermi_without_redshift]
Fermi_long_Epeak_in_keV			=	Fermi_Epeak[inds_long_in_Fermi_without_redshift]		# in keV.
Fermi_long_Epeak_in_keV_error	=	Fermi_Epeak_error[inds_long_in_Fermi_without_redshift]	# same as above.
Fermi_long_Epeak_in_MeV			=	1e-3 * Fermi_long_Epeak_in_keV			# in MeV.
Fermi_long_Epeak_in_MeV_error	=	1e-3 * Fermi_long_Epeak_in_keV_error	# same as above.
Fermi_long_alpha				=	Fermi_alpha[inds_long_in_Fermi_without_redshift]
Fermi_long_beta					=	Fermi_beta[inds_long_in_Fermi_without_redshift]

print '#### Fermi long GRBs ####', '\n'
print 'Number of GRBs put in		:	', Fermi_long_flux.size, '\n'
Fermi_long_name, Fermi_long_flux, Fermi_long_flux_error, Fermi_long_Epeak_in_MeV, Fermi_long_Epeak_in_MeV_error, Fermi_long_pseudo_redshift, Fermi_long_pseudo_redshift_error, Fermi_long_Luminosity, Fermi_long_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Fermi( Fermi_long_name, Fermi_long_flux, Fermi_long_flux_error, Fermi_long_Epeak_in_MeV, Fermi_long_Epeak_in_MeV_error, Fermi_long_alpha, Fermi_long_beta )
print 'Number of GRBs selected		:	', Fermi_long_flux.size

Fermi_short_name				=	Fermi_name[inds_short_in_Fermi_without_redshift]
Fermi_short_flux				=	Fermi_flux[inds_short_in_Fermi_without_redshift]
Fermi_short_flux_error			=	Fermi_flux_error[inds_short_in_Fermi_without_redshift]
Fermi_short_Epeak_in_keV		=	Fermi_Epeak[inds_short_in_Fermi_without_redshift]		# in keV.
Fermi_short_Epeak_in_keV_error	=	Fermi_Epeak_error[inds_short_in_Fermi_without_redshift]	# same as above.
Fermi_short_Epeak_in_MeV		=	1e-3 * Fermi_short_Epeak_in_keV			# in MeV.
Fermi_short_Epeak_in_MeV_error	=	1e-3 * Fermi_short_Epeak_in_keV_error	# same as above.
Fermi_short_alpha				=	Fermi_alpha[inds_short_in_Fermi_without_redshift]
Fermi_short_beta				=	Fermi_beta[inds_short_in_Fermi_without_redshift]

print '\n\n'
print '#### Fermi short GRBs ####', '\n'
print 'Number of GRBs put in		:	', Fermi_short_flux.size, '\n'
Fermi_short_name, Fermi_short_flux, Fermi_short_flux_error, Fermi_short_Epeak_in_MeV, Fermi_short_Epeak_in_MeV_error, Fermi_short_pseudo_redshift, Fermi_short_pseudo_redshift_error, Fermi_short_Luminosity, Fermi_short_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Fermi( Fermi_short_name, Fermi_short_flux, Fermi_short_flux_error, Fermi_short_Epeak_in_MeV, Fermi_short_Epeak_in_MeV_error, Fermi_short_alpha, Fermi_short_beta )
print 'Number of GRBs selected		:	', Fermi_short_flux.size
print '\n\n\n\n'


####################################################################################################################################################








####################################################################################################################################################
########	For the Swift GRBs, excluding those common with Fermi (since they have spectra), and those with known redshifts.




inds_common		=	choose( Swift_all_name, common_all_Swift_name )
inds_wkr		=	choose( Swift_all_name, Swift_wkr_name )
inds_to_delete	=	np.union1d( inds_common, inds_wkr )
inds_exclusively_Swift_GRBs_without_redshift	=	np.delete( np.arange(Swift_all_num), inds_to_delete )
Swift_num	=	inds_exclusively_Swift_GRBs_without_redshift.size

print '\n\n\n\n\n\n\n\n'
print ' #### Swift GRBs ####', '\n'
print '# of common GRBs		:	', inds_common.size
print '# of GRBs with redshift		:	', inds_wkr.size
print '# of common amongst these	:	', np.intersect1d( inds_common, inds_wkr ).size
print '# to be finally deleted		:	', inds_to_delete.size, '\n'
print '# of Swift GRBs, total		:	', Swift_all_num
print '# to be selected		:	', Swift_num

Swift_name			=	Swift_all_name[inds_exclusively_Swift_GRBs_without_redshift]
Swift_T90			=	Swift_all_T90[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux			=	Swift_all_flux[inds_exclusively_Swift_GRBs_without_redshift]
Swift_flux_error	=	Swift_all_flux_error[inds_exclusively_Swift_GRBs_without_redshift]

#	ascii.write(  Table( [Swift_name], names = ['Swift name'] ),  './../tables/Swift_only_names.txt', format = 'fixed_width', overwrite = True  )

#	print np.sort(Swift_flux)[0:20]
#	indices_to_delete	=	np.where(Swift_flux < 0.2)[0]
#	print indices_to_delete.size
#	Swift_all_name		=	np.delete( Swift_name      , indices_to_delete )
#	Swift_all_T90		=	np.delete( Swift_T90       , indices_to_delete )
#	Swift_all_flux		=	np.delete( Swift_flux      , indices_to_delete )
#	Swift_all_flux_error=	np.delete( Swift_flux_error, indices_to_delete )
#	Swift_all_num		=	Swift_all_name.size
#	print Swift_flux[0:20]




hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Fermi_Epeak), 0.125, 1, 4 )	;	hx	=	hist[0]	;	hy	=	hist[1]
fits	=	mf.fit_a_gaussian( hx, hy )	;	f0	=	fits[0]	;	f1	=	fits[1]	;	f2	=	fits[2]

Swift_Epeak_in_keV			=	np.random.normal( f0, f1, Swift_num )
Swift_Epeak_in_keV			=	10**Swift_Epeak_in_keV			#	in keV
Swift_Epeak_in_MeV			=	1e-3 * Swift_Epeak_in_keV		#	in MeV
#	Swift_Epeak_in_MeV_error	=	np.abs( Swift_Epeak_in_MeV - np.mean(Fermi_Epeak*1e-3) )
Swift_Epeak_in_MeV_error	=	np.zeros( Swift_Epeak_in_MeV.size )

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Swift_Epeak_in_keV), 0.125, 1, 4 )	;	sx	=	hist[0]	;	sy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/Fermi--Ep_distribution.png' )
plt.clf()
plt.close()

hist	=	mf.my_histogram_according_to_given_boundaries( np.log10(Swift_Epeak_in_keV), 0.125, 1, 4 )	;	sx	=	hist[0]	;	sy	=	hist[1]
plt.xlabel( r'$ \rm{ log( \, } $' + r'$ E_p $' + r'$ \rm{ \, [keV] \, ) } $' , fontsize = size_font )
plt.plot( hx, mf.gaussian(hx, f0, f1, f2), 'k-', label = r'$ Fermi \rm{ , \; fit } $' )
plt.step( hx, hy, color = 'r', label = r'$ Fermi \rm{ , \; data } $' )
plt.step( sx, sy * (hy.max()/sy.max()), color = 'b', label = r'$ Swift \rm{ , \; simulated } $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/pseudo_calculations/Swift--Ep_distribution--simulated_1.png' )
plt.clf()
plt.close()

plt.title( r'$ Swift $', fontsize = size_font )
plt.hist( Swift_Epeak_in_keV, bins = np.logspace(1, 4, 20) )
plt.gca().set_xscale('log')
plt.xlabel( r'$ E_p \; \rm{ [keV] } $', fontsize = size_font )
plt.savefig( './../plots/pseudo_calculations/Swift--Ep_distribution--simulated_2.png' )
plt.clf()
plt.close()





inds_long_in_Swift	=	np.where( Swift_T90 >= T90_cut )[0]
inds_short_in_Swift	=	np.delete( np.arange(Swift_num), inds_long_in_Swift )

Swift_long_name					=	Swift_name[inds_long_in_Swift]
Swift_long_T90					=	Swift_T90[inds_long_in_Swift]
Swift_long_flux					=	Swift_flux[inds_long_in_Swift]
Swift_long_flux_error			=	Swift_flux_error[inds_long_in_Swift]
Swift_long_Epeak_in_MeV			=	Swift_Epeak_in_MeV[inds_long_in_Swift]
Swift_long_Epeak_in_MeV_error	=	Swift_Epeak_in_MeV_error[inds_long_in_Swift]
Swift_long_num					=	Swift_long_name.size

Swift_short_name				=	Swift_name[inds_short_in_Swift]
Swift_short_T90					=	Swift_T90[inds_short_in_Swift]
Swift_short_flux				=	Swift_flux[inds_short_in_Swift]
Swift_short_flux_error			=	Swift_flux_error[inds_short_in_Swift]
Swift_short_Epeak_in_MeV		=	Swift_Epeak_in_MeV[inds_short_in_Swift]
Swift_short_Epeak_in_MeV_error	=	Swift_Epeak_in_MeV_error[inds_short_in_Swift]
Swift_short_num					=	Swift_short_name.size


print 'Out of which, # of long  GRBs	:	', Swift_long_num
print '                   short GRBs	:	', Swift_short_num



print '\n\n\n\n'
print '#### Swift long GRBs ####', '\n'
print 'Number of GRBs put in		:	', Swift_long_num, '\n'
Swift_long_name, Swift_long_flux, Swift_long_flux_error, Swift_long_Epeak_in_MeV, Swift_long_Epeak_in_MeV_error, Swift_long_pseudo_redshift, Swift_long_pseudo_redshift_error, Swift_long_Luminosity, Swift_long_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Swift( Swift_long_name, Swift_long_flux, Swift_long_flux_error, Swift_long_Epeak_in_MeV, Swift_long_Epeak_in_MeV_error )
print 'Number of GRBs selected		:	', Swift_long_flux.size

print '\n\n'
print '#### Swift short GRBs ####', '\n'
print 'Number of GRBs put in		:	', Swift_short_num, '\n'
Swift_short_name, Swift_short_flux, Swift_short_flux_error, Swift_short_Epeak_in_MeV, Swift_short_Epeak_in_MeV_error, Swift_short_pseudo_redshift, Swift_short_pseudo_redshift_error, Swift_short_Luminosity, Swift_short_Luminosity_error	=	estimate_pseudo_redshift_and_Luminosity__Swift( Swift_short_name, Swift_short_flux, Swift_short_flux_error, Swift_short_Epeak_in_MeV, Swift_short_Epeak_in_MeV_error )
print 'Number of GRBs selected		:	', Swift_short_flux.size
print '\n\n\n\n'



####################################################################################################################################################












####################################################################################################################################################
########	For the Swift-only GRBs with known redshifts, called "other" GRBs.




inds_in_Swift_wkr_for_common_wkr	=	choose( Swift_wkr_name, common_wkr_Swift_name )
inds_exclusively_Swift_GRBs_with_redshift	=	np.delete( np.arange(Swift_wkr_num), inds_in_Swift_wkr_for_common_wkr )
other_num	=	inds_exclusively_Swift_GRBs_with_redshift.size

print '\n\n\n\n\n\n\n\n'
print ' #### other GRBs ####', '\n'
print '# of Swift  GRBs wkr	:	', Swift_wkr_num
print '# of common GRBs wkr	:	', common_wkr_num
print '# of Swift-only wkr	:	', other_num

other_Swift_name			=	Swift_wkr_name[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_redshift		=	Swift_wkr_redhsift[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_T90				=	Swift_wkr_T90[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux			=	Swift_wkr_flux[inds_exclusively_Swift_GRBs_with_redshift]
other_Swift_flux_error		=	Swift_wkr_flux_error[inds_exclusively_Swift_GRBs_with_redshift]



#	print ( other_Swift_flux == 0 ).any()
#	indices_to_keep			=	np.where( other_Swift_flux != 0 )[0]
#	other_Swift_name		=	other_Swift_name[indices_to_keep]
#	other_Swift_redshift	=	other_Swift_redshift[indices_to_keep]
#	other_Swift_T90			=	other_Swift_T90[indices_to_keep]
#	other_Swift_flux		=	other_Swift_flux[indices_to_keep]
#	other_Swift_flux_error	=	other_Swift_flux_error[indices_to_keep]


other_Luminosity	=	other_Swift_redshift.copy()
for j, z in enumerate(other_Swift_redshift):
	ep		=	other_Swift_flux[j]
	other_Luminosity[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( ep, z )
other_Luminosity		=	other_Luminosity * (cm_per_Mpc**2) * erg_per_keV
other_Luminosity_error	=	other_Luminosity * (other_Swift_flux_error/other_Swift_flux)



inds_other_long		=	np.where( other_Swift_T90 >= T90_cut )[0]
inds_other_short	=	np.delete( np.arange(other_num), inds_other_long )

other_long_name					=	other_Swift_name[inds_other_long]
other_long_redshift				=	other_Swift_redshift[inds_other_long]
other_long_Luminosity			=	other_Luminosity[inds_other_long]
other_long_Luminosity_error		=	other_Luminosity_error[inds_other_long]
other_long_num					=	inds_other_long.size

other_short_name				=	other_Swift_name[inds_other_short]
other_short_redshift			=	other_Swift_redshift[inds_other_short]
other_short_Luminosity			=	other_Luminosity[inds_other_short]
other_short_Luminosity_error	=	other_Luminosity_error[inds_other_short]
other_short_num					=	inds_other_short.size


print 'Out of which, # of long	:	', other_long_num
print '                  short	:	', other_short_num
print '\n\n\n\n'



####################################################################################################################################################









####################################################################################################################################################
########	For the common GRBs with known redshifts.


inds_common_long	=	np.where( common_wkr_Swift_T90 >= T90_cut )[0]
inds_common_short	=	np.delete( np.arange(common_wkr_num), inds_common_long )
print '\n\n\n\n\n\n\n\n'
print ' #### common GRBs ####', '\n'
print common_wkr_Fermi_name[inds_common_short]

known_long_Luminosity	=	common_wkr_Luminosity[inds_common_long ]
known_short_Luminosity	=	common_wkr_Luminosity[inds_common_short]


####################################################################################################################################################





####################################################################################################################################################
########	Writing the data.


print '\n\n\n\n\n\n\n\n'



L_vs_z__known_long	=	Table( [ common_wkr_redshift[inds_common_long ], common_wkr_Epeak[inds_common_long ], common_wkr_Epeak_error[inds_common_long ],  known_long_Luminosity, common_wkr_Luminosity_error[inds_common_long ] ], names = [ 'measured z', 'Epeak (meas)', 'Epeak_error (meas)', 'Luminosity', 'Luminosity error' ] )
L_vs_z__known_short	=	Table( [ common_wkr_redshift[inds_common_short], common_wkr_Epeak[inds_common_short], common_wkr_Epeak_error[inds_common_short], known_short_Luminosity, common_wkr_Luminosity_error[inds_common_short] ], names = [ 'measured z', 'Epeak (meas)', 'Epeak_error (meas)', 'Luminosity', 'Luminosity error' ] )
L_vs_z__Fermi_long	=	Table( [  Fermi_long_name,  Fermi_long_pseudo_redshift,  Fermi_long_pseudo_redshift_error,  Fermi_long_Epeak_in_MeV*1e3,  Fermi_long_Epeak_in_MeV_error*1e3,  Fermi_long_Luminosity,  Fermi_long_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (meas) [keV]', 'Epeak_error (meas) [keV]', 'Luminosity', 'Luminosity_error' ] )
L_vs_z__Fermi_short	=	Table( [ Fermi_short_name, Fermi_short_pseudo_redshift, Fermi_short_pseudo_redshift_error, Fermi_short_Epeak_in_MeV*1e3, Fermi_short_Epeak_in_MeV_error*1e3, Fermi_short_Luminosity, Fermi_short_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (meas) [keV]', 'Epeak_error (meas) [keV]', 'Luminosity', 'Luminosity_error' ] )
L_vs_z__Swift_long	=	Table( [  Swift_long_name,  Swift_long_pseudo_redshift,  Swift_long_pseudo_redshift_error,  Swift_long_Epeak_in_MeV*1e3,  Swift_long_Epeak_in_MeV_error*1e3,  Swift_long_Luminosity,  Swift_long_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (siml) [keV]', 'Epeak_error (siml) [keV]', 'Luminosity', 'Luminosity_error' ] )
L_vs_z__Swift_short	=	Table( [ Swift_short_name, Swift_short_pseudo_redshift, Swift_short_pseudo_redshift_error, Swift_short_Epeak_in_MeV*1e3, Swift_short_Epeak_in_MeV_error*1e3, Swift_short_Luminosity, Swift_short_Luminosity_error], names = [ 'name', 'pseudo z', 'pseudo z_error', 'Epeak (siml) [keV]', 'Epeak_error (siml) [keV]', 'Luminosity', 'Luminosity_error' ] )
L_vs_z__other_long	=	Table( [ other_long_redshift,  other_long_Luminosity,  other_long_Luminosity_error], names = [ 'measured z', 'Luminosity', 'Luminosity_error' ] )
L_vs_z__other_short	=	Table( [other_short_redshift, other_short_Luminosity, other_short_Luminosity_error], names = [ 'measured z', 'Luminosity', 'Luminosity_error' ] )


ascii.write( L_vs_z__known_long , './../tables/L_vs_z__known_long.txt' , format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__known_short, './../tables/L_vs_z__known_short.txt', format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__Fermi_long , './../tables/L_vs_z__Fermi_long---with_names.txt' , format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__Fermi_short, './../tables/L_vs_z__Fermi_short--with_names.txt', format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__Swift_long , './../tables/L_vs_z__Swift_long---with_names.txt' , format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__Swift_short, './../tables/L_vs_z__Swift_short--with_names.txt', format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__other_long , './../tables/L_vs_z__other_long.txt' , format = 'fixed_width', overwrite = True )
ascii.write( L_vs_z__other_short, './../tables/L_vs_z__other_short.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################
