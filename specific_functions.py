from __future__ import division
from astropy.io import ascii
from astropy.table import Table
from scipy.integrate import quad
import debduttaS_functions as mf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################

padding		= 	8	# The padding of the axes labels.
size_font	= 	18	# The fontsize in the images.

P	=	np.pi		# Dear old pi!
C	=	2.998*1e5	# The speed of light in vacuum, in km.s^{-1}.
H_0	=	72			# Hubble's constant, in km.s^{-1}.Mpc^{-1}.
CC	=	0.73		# Cosmological constant.

z_min		=	1e-5
z_max		=	1e2
z_num		=	1e4

alpha_fix	=	-0.566
beta_fix	=	-2.823
Ec_fix		=	181.338	#	in keV.


####################################################################################################################################################



####################################################################################################################################################


def chi_MD(z):
	
	temp	=	2 * C / H_0
	temp	=	temp * ( 1 - (1+z)**(-0.5) )		#	in Mpc.
	
	return temp

def integrand_chi(a):
	
	return a**(-2.0) / np.sqrt(  CC + (1-CC)*(a**(-3.0))  )

def chi(z):
	
	temp	=	C / H_0
	temp	=	temp * 	quad( integrand_chi, 1/(1+z), 1 )[0]
	
	return temp										#	in Mpc.

def dA(z):
	
	return	chi(z) / (1+z)							#	in Mpc.

def dL(z):
	
	return	chi(z) * (1+z)							#	in Mpc.


#~ z_sim	=	np.linspace(z_min, z_max, z_num)
#~ d_chi	=	np.zeros( z_sim.size )
#~ for j, z in enumerate( z_sim ):
	#~ d_chi[j]	=	chi(z)
#~ d_A	=	d_chi / ( 1 + z_sim )
#~ d_L	=	d_chi * ( 1 + z_sim )
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \rm{ [Mpc] } $', fontsize = size_font )
#~ plt.loglog( z_sim, d_chi , lw = 2, color = 'b', alpha = 1.0, label = r'$  \chi $' )
#~ plt.loglog( z_sim,   d_A , lw = 2, color = 'g', alpha = 0.8, label = r'$ d_{A} $' )
#~ plt.loglog( z_sim,   d_L , lw = 2, color = 'r', alpha = 0.7, label = r'$ d_{L} $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/distance_vs_redshift--all.png' )
#~ plt.clf()
#~ plt.close()
#~ plt.xlabel( r'$ z $', fontsize = size_font+2 )
#~ plt.ylabel( r'$ \chi \, \rm{ [Mpc] } $', fontsize = size_font )
#~ plt.loglog( z_sim, chi_MD(z_sim) , lw = 2, color = 'k', linestyle = 'dashed', label = r'$  \Lambda = 0    $' )
#~ plt.loglog( z_sim,         d_chi , lw = 2, color = 'k', linestyle =  'solid', label = r'$  \Lambda = 0.73 $' )
#~ plt.legend( numpoints = 1, loc = 'best' )
#~ plt.savefig( './../plots/distance_vs_redshift--chi.png' )
#~ plt.clf()
#~ plt.close()




def dVc_by_onepluszee(z):
	
	temp	=	(C/H_0)
	temp	=	temp * chi(z)**2
	temp	=	temp / (1+z)
	temp	=	temp / np.sqrt(  CC + (1-CC)*( (1+z)**3 )  )
	
	return temp







def S( E, alpha, beta, E_c ):
	
	
	E_cutoff	=	(alpha - beta)*E_c
	if E <= E_cutoff:
		temp	=	(E**alpha) * np.exp( - E/E_c )
	else:
		temp	=	( ((alpha-beta)*E_c)**(alpha-beta) ) * ( E**beta ) * np.exp( -(alpha-beta) )
	
	return temp

#~ def GRB_band( E, alpha, beta, E_c ):
	#~ 
	#~ 
	#~ '''
	#~ 
	#~ 
	#~ Parameters
	#~ -----------
	#~ E:		Array containing values of Energy, in keV.
	#~ alpha:	1st power-law index.
	#~ beta:	2nd power-law index.
	#~ E_c:	Characteristic energy, in keV.
	#~ 
	#~ Returns
	#~ ----------
	#~ spec:	Normalized band-spectrum over the full array.
	#~ 
	#~ 
	#~ '''
	#~ 
	#~ 
	#~ E_cutoff	=	(alpha - beta)*E_c
	#~ E_lower 	=	E[ E <= E_cutoff ]
	#~ spec_lower	=	( (E_lower/100)**alpha ) * np.exp( - E_lower/E_c )
	#~ E_upper		=	E[ E >  E_cutoff ]
	#~ spec_upper	=	( ((alpha-beta)*(E_c/100))**(alpha-beta) ) * ( (E_upper/100)**beta ) * np.exp( -(alpha-beta) )
	#~ spec		=	np.append( spec_lower, spec_upper )
	#~ 
	#~ return spec

def integrand_k__known__photonflux( E, alpha, beta, E_c ):
	return E * S( E, alpha, beta, E_c )

def integrand_k__fixed__photonflux( E ):
	return E * S( E, alpha_fix, beta_fix, Ec_fix )

def integrand_k__known__countflux(  E, alpha, beta, E_c ):
	return S( E, alpha, beta, Ec )

def integrand_k__fixed__countflux(  E ):
	return S( E, alpha_fix, beta_fix, Ec_fix )


def k_correction_factor_with_known_spectral_parameters__GBM(alpha, beta, E_c, z):
	
	#	all energies in keV
	E_min		=	8
	E_max		=	1e4
		
	numerator	=	quad( integrand_k__known__photonflux, 1, 1e4, args = (alpha, beta, E_c) )[0]
	denominator	=	quad( integrand_k__known__photonflux, (1+z)*E_min, (1+z)*E_max, args = (alpha, beta, E_c) )[0]
	
	temp		=	numerator / denominator
	
	return temp

def k_correction_factor_with_known_spectral_parameters__BAT(alpha, beta, E_c, z):
	
	#	all energies in keV
	E_min		=	15
	E_max		=	150
		
	numerator	=	quad( integrand_k__known__photonflux, 1, 1e4, args = (alpha, beta, E_c) )[0]
	denominator	=	quad( integrand_k__known__countflux , (1+z)*E_min, (1+z)*E_max, args = (alpha, beta, E_c) )[0]
	
	temp		=	numerator / denominator
	
	return temp

def k_correction_factor_with_fixed_spectral_parameters__GBM(z):
	
	#	all energies in keV
	E_min		=	8
	E_max		=	1e4
	
	numerator	=	quad( integrand_k__fixed__photonflux, 1, 1e4 )[0]
	denominator	=	quad( integrand_k__fixed__photonflux, (1+z)*E_min, (1+z)*E_max )[0]
	
	temp		=	numerator / denominator
	
	return temp

def k_correction_factor_with_fixed_spectral_parameters__BAT(z):
	
	#	all energies in keV
	E_min		=	15
	E_max		=	150
	
	numerator	=	quad( integrand_k__fixed__photonflux, 1, 1e4 )[0]
	denominator	=	quad( integrand_k__fixed__countflux , (1+z)*E_min, (1+z)*E_max )[0]
	
	temp		=	numerator / denominator
	
	return temp


def Liso_with_known_spectral_parameters__Fermi( ef, alpha, beta, E_c, z ):
	
	return ef * 4*P * dL(z)**2 * k_correction_factor_with_known_spectral_parameters__GBM(alpha, beta, E_c, z)

def Liso_with_known_spectral_parameters__Swift( ef, alpha, beta, E_c, z ):
	
	return ef * 4*P * dL(z)**2 * k_correction_factor_with_known_spectral_parameters__BAT(alpha, beta, E_c, z)

def Liso_with_fixed_spectral_parameters__Fermi( ef, z ):
	
	return ef * 4*P * dL(z)**2 * k_correction_factor_with_fixed_spectral_parameters__GBM(z)

def Liso_with_fixed_spectral_parameters__Swift( ef, z ):
	
	return ef * 4*P * dL(z)**2 * k_correction_factor_with_fixed_spectral_parameters__BAT(z)





def k_correction_factor_with_fixed_spectral_parameters__CZT(z):
	
	#	all energies in keV
	E_min		=	20
	E_max		=	200
	
	numerator	=	quad( integrand_k__fixed__photonflux, 1, 1e4 )[0]
	denominator	=	quad( integrand_k__fixed__countflux , (1+z)*E_min, (1+z)*E_max )[0]
	
	temp		=	numerator / denominator
	
	return temp

def Liso_with_fixed_spectral_parameters__CZTI( ef, z ):
	
	return ef * 4*P * dL(z)**2 * k_correction_factor_with_fixed_spectral_parameters__CZT(z)






#~ def k_correction_factor_with_known_spectral_parameters__GBM__with_errors(alpha, alpha_error, beta, beta_error, E_c, E_c_error, z):
	#~ 
	#~ #	all energies in keV
	#~ E_min		=	8
	#~ E_max		=	1e4
	#~ 
	#~ N0	=	quad( integrand_k__known__photonflux,           1,         1e4, args = (alpha, beta, E_c) )[0]
	#~ D0	=	quad( integrand_k__known__photonflux, (1+z)*E_min, (1+z)*E_max, args = (alpha, beta, E_c) )[0]
	#~ 
	#~ first_term	=	(   quad( integrand_k__known__photonflux,           1,         1e4, args = (alpha+alpha_error, beta, E_c) )[0]  +  quad( integrand_k_with_known_spectral_parameters,           1,         1e4, args = (alpha, beta+beta_error, E_c) )[0]  +  quad( integrand_k_with_known_spectral_parameters,           1,         1e4, args = (alpha, beta, E_c+E_c_error) )[0]   )/N0 - 3
	#~ second_term	=	(   quad( integrand_k__known__photonflux, (1+z)*E_min, (1+z)*E_max, args = (alpha+alpha_error, beta, E_c) )[0]  +  quad( integrand_k_with_known_spectral_parameters, (1+z)*E_min, (1+z)*E_max, args = (alpha, beta+beta_error, E_c) )[0]  +  quad( integrand_k_with_known_spectral_parameters, (1+z)*E_min, (1+z)*E_max, args = (alpha, beta, E_c+E_c_error) )[0]   )/D0 - 3
	#~ 
	#~ k			=	N0 / D0
	#~ k_error		=	k * (first_term + second_term)
	#~ 
	#~ return k, k_error
#~ 
#~ def Liso_with_known_spectral_parameters__Fermi__with_errors( ef, ef_err, alpha, alpha_error, beta, beta_error, E_c, E_c_error, z ):
	#~ 
	#~ temp	=	k_correction_factor_with_known_spectral_parameters__GBM__with_errors(alpha, alpha_error, beta, beta_error, E_c, E_c_error, z)
	#~ k		=	temp[0]
	#~ k_err	=	temp[1]
	#~ 
	#~ L		=	ef * 4*P * dL(z)**2 * k
	#~ L_error	=	L * (  (ef_err/ef) + (k_err/k)  )
	#~ return L, L_error












def my_histogram_with_errorbars( array, array_poserr, array_negerr, x_bin, x_min, x_max ):
	
	
	'''
	
	
	Parameters
	-----------
	array		:	The array to be binned.
	array_poserr:	The positive errors on the array.
	array_negerr:	The negative errors on the array.
	x_bin		:	The bin of the "x" axis.
	x_min		:	The minimum of the binning axis.
	x_max		:	The maximum of the binning axis.
	
	Returns
	-----------
	x_mid		:	The x-axis of the histogram (middle of the bins).
	y			:	The y-axis of the histogram.
	y_poserr	:	The positive errors along the y-axis.
	y_negerr	:	The negative errors along the y-axis.
	
	
	'''
	
	
	array_min	=	array - array_negerr
	array_max	=	array + array_poserr
	x_left	=	np.arange( x_min, x_max, x_bin )
	x_mid	=	x_left + x_bin/2
	
	y			=	np.zeros( x_left.size )
	y_poserr	=	np.zeros( x_left.size )
	y_negerr	=	np.zeros( x_left.size )
	
	for j, left in enumerate(x_left):
		right	=	left + x_bin
		
		pos_error	=	0
		neg_error	=	0
		
		
		ind		=	np.where( (left<= array) & (array<right) )[0]
		y[j]	=	ind.size
		
		array_lower		=	array_min[ind]
		neg_error		+=	np.where( array_lower < left )[0].size
		array_upper		=	array_max[ind]
		neg_error		+=	np.where( array_lower > right )[0].size
		y_negerr[j]		=	neg_error
		
		ind_greater		=	np.where( array > right )[0]
		array_greater	=	array_min[ind_greater]
		pos_error		+=	np.where( (left<= array_greater) & (array_greater<right) )[0].size
		ind_smaller		=	np.where( array < left )[0]
		array_smaller	=	array_max[ind_smaller]
		pos_error		+=	np.where( (left<= array_smaller) & (array_smaller<right) )[0].size
		y_poserr[j]		=	pos_error
		
		
	return x_mid, y, y_poserr, y_negerr
