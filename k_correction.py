from __future__ import division
from astropy.io import ascii
from astropy.table import Table
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

padding		= 	18	# The padding of the axes labels.
size_font	= 	18	# The fontsize in the images.
marker_size	=	10	# The size of markers in scatter plots.

P	=	np.pi		# Dear old pi!

z_min		=	1e-5
z_max		=	1e2
z_bin		=	1e5


alpha_fix	=	-0.566
beta_fix	=	-2.823
Ec_fix		=	181.338	#	in keV.


####################################################################################################################################################







####################################################################################################################################################


##	For the GRBs with known spectral parameters as well as redshifts.

common_GRBs_table	=	ascii.read( './../tables/common_GRBs--wkr.txt', format = 'fixed_width' )
common_redshift		=	common_GRBs_table['redshift'].data
common_Epeak		=	common_GRBs_table['Epeak'].data
common_alpha		=	common_GRBs_table['alpha'].data
common_beta			=	common_GRBs_table['beta'].data
common_k			=	common_redshift.copy()
for j, z in enumerate(common_redshift):
	alpha			=	common_alpha[j]
	beta			=	common_beta[j]
	E_c				=	common_Epeak[j]
	common_k[j]		=	sf.k_correction_factor_with_known_spectral_parameters__GBM(alpha, beta, E_c, z)

k_table	=	Table( [common_redshift, common_k], names = ['z', 'k'] )
ascii.write( k_table, './../tables/k_common_GRBs__wkr--all.txt', format = 'fixed_width', overwrite = True )

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ k } $', fontsize = size_font+2, rotation = 0, labelpad = padding )
ax.plot( common_redshift, common_k, 'k.', ms = marker_size )
plt.savefig( './../plots/k_Fermi--common_GRBs__wkr--all.png' )
plt.clf()
plt.close()


####################################################################################################################################################






####################################################################################################################################################


#~ ##	To create a template of k-corrections for average spectral parameters, derived from the actual Fermi sample.
#~ 
#~ z_sim	=	np.linspace(z_min, z_max, z_bin)
#~ t0	=	time.time()
#~ def integral_3rdterm__Fermi( E ):
	#~ return ( E**2 ) * sf.S( E, alpha_fix, beta_fix, Ec_fix )
#~ def integrand_3rdterm__Fermi( E ):
	#~ return E * sf.S( E, alpha_fix, beta_fix, Ec_fix )
#~ def integral_3rdterm__Swift( E ):
	#~ return E * sf.S( E, alpha_fix, beta_fix, Ec_fix )
#~ def integrand_3rdterm__Swift( E ):
	#~ return sf.S( E, alpha_fix, beta_fix, Ec_fix )
#~ 
#~ 
#~ 
#~ E_min		=	8		#	in keV, Fermi band lower energy.
#~ E_max		=	1e4		#	in keV, Fermi band upper energy.
#~ d_chi		=	np.zeros( z_sim.size )
#~ term_Fermi	=	np.zeros( z_sim.size )
#~ for j, z in enumerate( z_sim ):
	#~ d_chi[j]		=	sf.chi(z)
	#~ term_Fermi[j]	=	quad( integrand_3rdterm__Fermi, (1+z)*E_min, (1+z)*E_max )[0]
#~ dL_sim		=	d_chi * ( 1 + z_sim )
#~ term_Fermi	=	( integral_3rdterm__Fermi(E_min) + integral_3rdterm__Fermi(E_max) ) / term_Fermi
#~ print 'Fermi distance and error terms done in mins:	' , ( time.time() - t0 ) / 60
#~ 
#~ E_min		=	15		#	in keV, Swift band lower energy.
#~ E_max		=	150		#	in keV, Swift band upper energy.
#~ term_Swift	=	np.zeros( z_sim.size )
#~ for j, z in enumerate( z_sim ):
	#~ term_Swift[j]	=	quad( integrand_3rdterm__Swift, (1+z)*E_min, (1+z)*E_max )[0]
#~ term_Swift	=	( integral_3rdterm__Swift(E_min) + integral_3rdterm__Swift(E_max) ) / term_Swift
#~ print 'Swift distance and error terms done in mins:	' , ( time.time() - t0 ) / 60, '\n'
#~ 
#~ 
#~ k_Fermi_sim	=	z_sim.copy()
#~ for j, z in enumerate(z_sim):
	#~ k_Fermi_sim[j]	=	sf.k_correction_factor_with_fixed_spectral_parameters__GBM(z)
#~ print 'Fermi k done in mins:				' , ( time.time() - t0 ) / 60
#~ 
#~ k_Swift_sim	=	z_sim.copy()
#~ for j, z in enumerate(z_sim):
	#~ k_Swift_sim[j]	=	sf.k_correction_factor_with_fixed_spectral_parameters__BAT(z)
#~ print 'Swift k done in mins:				' , ( time.time() - t0 ) / 60
#~ 
#~ k_CZTI_sim	=	z_sim.copy()
#~ for j, z in enumerate(z_sim):
	#~ k_CZTI_sim[j]	=	sf.k_correction_factor_with_fixed_spectral_parameters__CZT(z)
#~ print 'CZTI  k done in mins:				' , ( time.time() - t0 ) / 60
#~ 
#~ k_table	=	Table( [z_sim, dL_sim, k_Fermi_sim, k_Swift_sim, term_Fermi, term_Swift, k_CZTI_sim], names = ['z', 'dL', 'k_Fermi', 'k_Swift', 'term_Fermi', 'term_Swift', 'k_CZTI'] )
#~ ascii.write( k_table, './../tables/k_table.txt', format = 'fixed_width', overwrite = True )


k_table	=	ascii.read( './../tables/k_table.txt', format = 'fixed_width' )
z_sim	=	k_table['z'].data
k_Fermi	=	k_table['k_Fermi'].data
k_Swift	=	k_table['k_Swift'].data
k_CZTI	=	k_table['k_CZTI' ].data

#~ ax	=	plt.subplot(111)
#~ ax.set_xscale('log')
#~ ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
#~ ax.set_ylabel( r'$ \rm{ k } $', fontsize = size_font+2, rotation = 0, labelpad = padding )
#~ ax.plot( z_sim, k_Fermi, 'k-', ms = marker_size )
#~ ax.plot( z_sim, np.ones(z_sim.size), 'k--', ms = marker_size )
#~ plt.savefig( './../plots/k_correction--Fermi.png' )
#~ plt.clf()
#~ plt.close()

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ k } $', fontsize = size_font+2, rotation = 0, labelpad = padding-8 )
ax.plot( z_sim, k_Fermi, 'k-', ms = marker_size )
ax.plot( z_sim, np.ones(z_sim.size), 'k--', ms = marker_size )
plt.savefig( './../plots/k_correction--Fermi.png' )
plt.clf()
plt.close()


ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ k \; [ } 210 \rm{ \, keV ] } $', fontsize = size_font, labelpad = padding-8 )
ax.plot( z_sim, k_Swift/210, 'k-', ms = marker_size )
ax.plot( z_sim, np.ones(z_sim.size), 'k--', ms = marker_size )
plt.savefig( './../plots/k_correction--Swift.png' )
plt.clf()
plt.close()

ax	=	plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ z $', fontsize = size_font+2 )
ax.set_ylabel( r'$ \rm{ k \, [keV] } $', fontsize = size_font, labelpad = padding-8 )
ax.plot( z_sim, k_Swift, 'b-', ms = marker_size, label = r'$ Swift $' )
ax.plot( z_sim, k_CZTI , 'g--', ms = marker_size, label = r'$ CZTI  $' )
plt.legend( numpoints = 1, loc = 'best' )
plt.savefig( './../plots/k_correction--Swift_and_CZTI.png' )
plt.clf()
plt.close()
