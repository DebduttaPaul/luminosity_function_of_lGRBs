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
plt.rc('font', family = 'serif', serif = 'cm10', size = 15)
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
#	BAT_sensitivity	=	0.04		# in  ph.s^{-1}.cm^{2}, smaller, for test: test fails badly.
CZT_sensitivity	=	0.20		# in  ph.s^{-1}.cm^{2}.


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


def straight_line( x, m, c ):
	return m*x + c


def fit_the_trend( x, y ):
	
	r, p_r	=	R( x, y )
	s, p_s	=	S( x, y )
	t, p_t	=	T( x, y )
	print 'R, p			:	', r, p_r
	print 'S, p			:	', s, p_s
	print 'T, p			:	', t, p_t
	
	x_to_fit_log	=	np.log10( x )
	y_to_fit_log	=	np.log10( y )
	
	#	plt.scatter( x_to_fit_log, y_to_fit_log )
	#	plt.show()
	
	popt, pcov		=	curve_fit( straight_line, x_to_fit_log, y_to_fit_log )
	exponent		=	popt[0]		;	coefficient			=	10**popt[1] *L_norm
	exponent_error	=	pcov[0,0]	;	coefficient_error	=	( 10**pcov[1,1] - 1 ) * coefficient
	print 'Coefficient		:	',	round(coefficient, 3), round(coefficient_error, 5)
	print 'Exponent		:	',		round(   exponent, 3), round(   exponent_error, 3)
	print '\n\n\n\n'
	
	return coefficient, coefficient_error, exponent, exponent_error


####################################################################################################################################################



#~ ####################################################################################################################################################
#~ 
#~ 
#~ k_table		=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
#~ z_sim		=	k_table['z'].data
#~ 
#~ L_cut__CZTI		=	z_sim.copy()
#~ L_cut__Fermi	=	z_sim.copy()
#~ L_cut__Swift	=	z_sim.copy()
#~ 
#~ t0	=	time.time()
#~ 
#~ 
#~ print 'Started calculating Fermi sensitivity limits...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__Fermi[j]	=	sf.Liso_with_fixed_spectral_parameters__Fermi( GBM_sensitivity, z )
#~ L_cut__Fermi	=	L_cut__Fermi * (cm_per_Mpc**2)
#~ print 'Done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
#~ 
#~ print 'Started calculating Swift sensitivity limits...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__Swift[j]	=	sf.Liso_with_fixed_spectral_parameters__Swift( BAT_sensitivity, z )
#~ L_cut__Swift	=	L_cut__Swift * (cm_per_Mpc**2) * erg_per_keV
#~ 
#~ print 'Done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 ), '\n'
#~ 
#~ 
#~ print 'Started calculating CZTI  sensitivity limits...'
#~ for j, z in enumerate(z_sim):
	#~ L_cut__CZTI[j]	=	sf.Liso_with_fixed_spectral_parameters__CZTI( CZT_sensitivity, z )
#~ L_cut__CZTI		=	L_cut__CZTI  * (cm_per_Mpc**2) * erg_per_keV
#~ print 'Done in {:.3f} mins.'.format( ( time.time() - t0 ) / 60 )
#~ print '\n\n\n'
#~ 
#~ ax	=	plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_yscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ L_{cut} \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font )
#~ ax.plot( z_sim, L_cut__Fermi, color = 'r', label = r'$ Fermi $'   )
#~ ax.plot( z_sim, L_cut__Swift, color = 'b', label = r'$ Swift $'   )
#~ ax.plot( z_sim, L_cut__CZTI , color = 'g', label = r'$\rm{CZTI}$' )
#~ plt.legend( loc = 'upper left' )
#~ plt.savefig('./../plots/pseudo_calculations/sensitivity_plot.png')
#~ plt.clf()
#~ plt.close()
#~ 
#~ 
#~ threshold_data	=	Table( [z_sim, L_cut__Fermi, L_cut__Swift, L_cut__CZTI], names = ['z_sim', 'L_cut__Fermi', 'L_cut__Swift', 'L_cut__CZTI'] )
#~ ascii.write( threshold_data, './../tables/thresholds.txt', format = 'fixed_width', overwrite = True )
#~ #	ascii.write( threshold_data, './../tables/thresholds--smaller.txt', format = 'fixed_width', overwrite = True )
#~ 
#~ 
#~ ####################################################################################################################################################





####################################################################################################################################################


threshold_data	=	ascii.read( './../tables/thresholds.txt', format = 'fixed_width' )
#	threshold_data	=	ascii.read( './../tables/thresholds--smaller.txt', format = 'fixed_width' )
z_sim			=	threshold_data['z_sim'].data
L_cut__Fermi	=	threshold_data['L_cut__Fermi'].data
L_cut__Swift	=	threshold_data['L_cut__Swift'].data
L_cut__CZTI		=	threshold_data['L_cut__CZTI' ].data


L_vs_z__known_long 	=	ascii.read( './../tables/L_vs_z__known_long.txt' , format = 'fixed_width' )
L_vs_z__known_short	=	ascii.read( './../tables/L_vs_z__known_short.txt', format = 'fixed_width' )
L_vs_z__Fermi_long 	=	ascii.read( './../tables/L_vs_z__Fermi_long.txt' , format = 'fixed_width' )
L_vs_z__Fermi_short	=	ascii.read( './../tables/L_vs_z__Fermi_short.txt', format = 'fixed_width' )
L_vs_z__Swift_long 	=	ascii.read( './../tables/L_vs_z__Swift_long.txt' , format = 'fixed_width' )
L_vs_z__Swift_short	=	ascii.read( './../tables/L_vs_z__Swift_short.txt', format = 'fixed_width' )
L_vs_z__other_long 	=	ascii.read( './../tables/L_vs_z__other_long.txt' , format = 'fixed_width' )
L_vs_z__other_short	=	ascii.read( './../tables/L_vs_z__other_short.txt', format = 'fixed_width' )

known_long_redshift			=	L_vs_z__known_long[ 'measured z'].data
known_long_Luminosity		=	L_vs_z__known_long[ 'Luminosity'].data       * L_norm
known_long_Luminosity_error	=	L_vs_z__known_long[ 'Luminosity error'].data * L_norm
known_short_redshift		=	L_vs_z__known_short['measured z'].data
known_short_Luminosity		=	L_vs_z__known_short['Luminosity'].data       * L_norm
known_short_Luminosity_error=	L_vs_z__known_short['Luminosity error'].data * L_norm

Fermi_long_redshift			=	L_vs_z__Fermi_long[  'pseudo z' ].data
Fermi_long_redshift_error	=	L_vs_z__Fermi_long[  'pseudo z_error' ].data
Fermi_long_Luminosity		=	L_vs_z__Fermi_long[ 'Luminosity'].data
Fermi_long_Luminosity_error	=	L_vs_z__Fermi_long[ 'Luminosity_error'].data
Fermi_short_redshift		=	L_vs_z__Fermi_short[ 'pseudo z' ].data
Fermi_short_redshift_error	=	L_vs_z__Fermi_short[ 'pseudo z_error' ].data
Fermi_short_Luminosity		=	L_vs_z__Fermi_short['Luminosity'].data
Fermi_short_Luminosity_error=	L_vs_z__Fermi_short['Luminosity_error'].data

Swift_long_redshift			=	L_vs_z__Swift_long[  'pseudo z' ].data
Swift_long_redshift_error	=	L_vs_z__Swift_long[  'pseudo z_error' ].data
Swift_long_Luminosity		=	L_vs_z__Swift_long[ 'Luminosity'].data
Swift_long_Luminosity_error	=	L_vs_z__Swift_long[ 'Luminosity_error'].data
Swift_short_redshift		=	L_vs_z__Swift_short[ 'pseudo z' ].data
Swift_short_redshift_error	=	L_vs_z__Swift_short[ 'pseudo z_error' ].data
Swift_short_Luminosity		=	L_vs_z__Swift_short['Luminosity'].data
Swift_short_Luminosity_error=	L_vs_z__Swift_short['Luminosity'].data

other_long_redshift			=	L_vs_z__other_long[ 'measured z'].data
other_long_Luminosity		=	L_vs_z__other_long[ 'Luminosity'].data
other_long_Luminosity_error	=	L_vs_z__other_long[ 'Luminosity_error'].data
other_short_redshift		=	L_vs_z__other_short['measured z'].data
other_short_Luminosity		=	L_vs_z__other_short['Luminosity'].data
other_short_Luminosity_error=	L_vs_z__other_short['Luminosity_error'].data

indices_to_keep				=	np.where( other_long_Luminosity != 0 )[0]
other_long_redshift			=	other_long_redshift[indices_to_keep]
other_long_Luminosity		=	other_long_Luminosity[indices_to_keep]
other_long_Luminosity_error	=	other_long_Luminosity_error[indices_to_keep]



ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( known_long_redshift , known_long_Luminosity , fmt = '.', color = 'k', label = r'$ \rm{ known, long  } $' )
ax.errorbar( known_short_redshift, known_short_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ known, short } $' )
ax.plot( z_sim, L_cut__Fermi, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--known.png' )
plt.clf()
plt.close()

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ pseudo \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( Fermi_long_redshift , Fermi_long_Luminosity , fmt = '.', color = 'k', label = r'$ \rm{ Fermi, long  } $' )
ax.errorbar( Fermi_short_redshift, Fermi_short_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ Fermi, short } $' )
ax.plot( z_sim, L_cut__Fermi, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Fermi.png' )
plt.clf()
plt.close()

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ pseudo \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( Swift_long_redshift , Swift_long_Luminosity , fmt = '.', color = 'k', label = r'$ \rm{ Swift, long  } $' )
ax.errorbar( Swift_short_redshift, Swift_short_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ Swift, short } $' )
ax.plot( z_sim, L_cut__Swift, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Swift.png' )
plt.clf()
plt.close()

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ measured \; \, redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( other_long_redshift , other_long_Luminosity , fmt = '.', color = 'k', label = r'$ \rm{ other, long  } $' )
ax.errorbar( other_short_redshift, other_short_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ other, short } $' )
ax.plot( z_sim, L_cut__Swift, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--other.png' )
plt.clf()
plt.close()








ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( known_long_redshift, known_long_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ Fermi, known  } $' )
ax.errorbar( Fermi_long_redshift, Fermi_long_Luminosity, fmt = '.', color = 'k', label = r'$ \rm{ Fermi, pseudo } $' )
ax.plot( z_sim, L_cut__Fermi, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Fermi_long_all.png' )
plt.clf()
plt.close()

print 'Fermi....\n'
#~ print np.median(   Fermi_long_redshift_error /  Fermi_long_redshift  )*100
#~ print np.median( known_long_Luminosity_error / known_long_Luminosity )*100, np.median( Fermi_long_Luminosity_error / Fermi_long_Luminosity )*100
print np.mean(   Fermi_long_redshift_error /  Fermi_long_redshift  )*100
print np.mean( known_long_Luminosity_error / known_long_Luminosity )*100, np.mean( Fermi_long_Luminosity_error / Fermi_long_Luminosity )*100
print '\n'

Fermi_long_redshift__all	=	np.append( known_long_redshift  , Fermi_long_redshift   )
Fermi_long_Luminosity__all	=	np.append( known_long_Luminosity, Fermi_long_Luminosity )
fit_the_trend( Fermi_long_redshift, Fermi_long_Luminosity/L_norm )









inds_to_delete	=	[]
for j, z in enumerate( Swift_long_redshift ):
	array	=	np.abs( z_sim - z )
	ind		=	np.where( array == array.min() )[0]
	if ( Swift_long_Luminosity[j] - L_cut__Swift[ind] ) < 0 :
		inds_to_delete.append( j )
inds_to_delete	=	np.array( inds_to_delete )
print 'Swift GRBs below threshold, deleted:	', inds_to_delete.size, '\n\n'
Swift_long_redshift			=	np.delete( Swift_long_redshift        , inds_to_delete )
Swift_long_redshift_error	=	np.delete( Swift_long_redshift_error  , inds_to_delete )
Swift_long_Luminosity		=	np.delete( Swift_long_Luminosity      , inds_to_delete )
Swift_long_Luminosity_error	=	np.delete( Swift_long_Luminosity_error, inds_to_delete )

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ \rm{ redshift } $', fontsize = size_font )
ax.set_ylabel( r'$ L_p \; $' + r'$ \rm{ [erg.s^{-1}] } $', fontsize = size_font, labelpad = padding-6 )
ax.set_xlim( z_min, z_max )
ax.set_ylim( y_in_eps_min, y_in_eps_max )
ax.errorbar( other_long_redshift, other_long_Luminosity, fmt = '.', color = 'r', label = r'$ \rm{ Swift, known  } $' )
ax.errorbar( Swift_long_redshift, Swift_long_Luminosity, fmt = '.', color = 'k', label = r'$ \rm{ Swift, pseudo } $' )
ax.plot( z_sim, L_cut__Swift, linestyle = '--', color = 'k' )
plt.legend( numpoints = 1, loc = 'upper left' )
plt.savefig( './../plots/pseudo_calculations/L_vs_z--Swift_long_all.png' )
plt.clf()
plt.close()

print 'Swift...\n'
#~ print np.median(   Swift_long_redshift_error /   Swift_long_redshift )*100
#~ print np.median( Swift_long_Luminosity_error / Swift_long_Luminosity )*100, np.median( other_long_Luminosity_error / other_long_Luminosity )*100
print np.mean(   Swift_long_redshift_error /   Swift_long_redshift )*100
print np.mean( Swift_long_Luminosity_error / Swift_long_Luminosity )*100, np.mean( other_long_Luminosity_error / other_long_Luminosity )*100
print '\n'

Swift_long_redshift__all	=	np.append( other_long_redshift  , Swift_long_redshift   )
Swift_long_Luminosity__all	=	np.append( other_long_Luminosity, Swift_long_Luminosity )
fit_the_trend( Swift_long_redshift, Swift_long_Luminosity/L_norm )
