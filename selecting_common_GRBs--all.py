from __future__ import division
from astropy.io import ascii
from astropy.table import Table
import debduttaS_functions as mf
import specific_functions as sf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', linewidth = 2)
plt.rc('font', family = 'serif', serif = 'cm10')
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']



####################################################################################################################################################


padding		= 	8	# The padding of the axes labels.
size_font	= 	16	# The fontsize in the images.
marker_size	=	10	# The size of markers in scatter plots.

P	=	np.pi		# Dear old pi!

L_norm		=	1e52	#	in ergs.s^{-1}.

cm_per_Mpc	=	3.0857 * 1e24
erg_per_keV	=	1.6020 * 1e-9


####################################################################################################################################################





####################################################################################################################################################


def convert( mother, mins, secs ):
	return ( mother*(60**2) + mins*60 + secs ) / (60**2)	# in hours


####################################################################################################################################################






####################################################################################################################################################


Swift_all_data			=	ascii.read( './../data/Swift--all_GRBs.txt' )
Swift_all_GRB_name		=	Swift_all_data['GRB'].data
Swift_all_Ttimes		=	Swift_all_data['Time [UT]'].data
Swift_all_RA			=	Swift_all_data['BAT RA (J2000)'].data
Swift_all_Dec			=	Swift_all_data['BAT Dec (J2000)'].data
Swift_all_error_radius	=	Swift_all_data['BAT 90% Error Radius [arcmin]'].data * (0.68/0.90)		#	in arcmin.
Swift_all_T90			=	Swift_all_data['BAT T90 [sec]'].data									#	in sec.
Swift_all_phoflux		=	Swift_all_data['BAT 1-sec Peak Photon Flux (15-150 keV) [ph/cm^2/sec]'].data								# in ph.cm^{-2}.s^{-1}.
Swift_all_phoflux_error	=	Swift_all_data['BAT 1-sec Peak Photon Flux 90% Error (15-150 keV) [ph/cm^2/sec]'].data * (0.68/0.90)		# in ph.cm^{-2}.s^{-1}.
Swift_all_num			=	Swift_all_GRB_name.size
Swift_all_RA			=	Swift_all_RA * 24/360
Swift_all_RA			=	np.round( Swift_all_RA, 3 )
Swift_all_error_radius	=	np.round( Swift_all_error_radius , 3 )	# in arcmin.
Swift_all_phoflux_error	=	np.round( Swift_all_phoflux_error, 11 )	# in erg.cm^{-2}.s^{-1}.

Swift_all_GRB_ID	=	np.zeros( Swift_all_num )
Swift_all_Tt		=	np.zeros( Swift_all_num )
for j, Ttime in enumerate(Swift_all_Ttimes):
	
	name		=	Swift_all_GRB_name[j]
	name		=	name[0:7]
	if len(name) == 7:
		ID = name[:-1]
	else:
		ID = name
	Swift_all_GRB_ID[j] = ID
		
	hour			=	float( Ttime[0:2] )
	mins			=	float( Ttime[3:5] )
	secs			=	float( Ttime[6:8] )
	decimal			=	convert( hour, mins, secs )
	Swift_all_Tt[j]	=	decimal
	
Swift_all_GRB_ID		=	Swift_all_GRB_ID.astype(int)
Swift_all_Tt			=	np.round( Swift_all_Tt , 3 )
print 'Swift GRBs with or without redshift measure	 	: ', Swift_all_GRB_name.size, '\n\n'

Swift_GRBs_table	=	Table( [ Swift_all_GRB_name, Swift_all_GRB_ID, Swift_all_Tt, Swift_all_RA, Swift_all_Dec, Swift_all_error_radius, Swift_all_T90, Swift_all_phoflux, Swift_all_phoflux_error ],
								names = [ 'Swift name', 'Swift ID', 'BAT Trigger-time', 'BAT RA', 'BAT Dec', 'BAT Error-radius', 'BAT T90', 'BAT Phoflux', 'BAT Phoflux_error' ] )
ascii.write( Swift_GRBs_table, './../tables/Swift_GRBs--all.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################






####################################################################################################################################################


Fermi_data				=	ascii.read( './../data/Fermi--all_GRBs.txt', format = 'fixed_width' )
Fermi_GRB_name			=	Fermi_data['name'].data
Fermi_Ttimes			=	Fermi_data['trigger_time'].data
Fermi_RAs				=	Fermi_data['ra'].data						#	in  hr,min,sec.
Fermi_Decs				=	Fermi_data['dec'].data						#	in deg,min,sec.
Fermi_error_radius		=	Fermi_data['error_radius'].data				#	in degree.
Fermi_T90				=	Fermi_data['t90'].data						#	in sec.
Fermi_T90_error			=	Fermi_data['t90_error'].data				#	in sec.
Fermi_fluence			=	Fermi_data['fluence'].data					#	in ergs.cm^{-2}.
Fermi_fluence_error		=	Fermi_data['fluence_error'].data			#	same as above.
Fermi_amp				=	Fermi_data['pflx_band_ampl'].data			#	in photons.cm^{-2}.s^{-1}.
Fermi_amp_pos_error		=	Fermi_data['pflx_band_ampl_pos_err'].data	#	same as above.
Fermi_amp_neg_error		=	Fermi_data['pflx_band_ampl_neg_err'].data	#	same as above.
Fermi_amp_error			=	np.maximum( Fermi_amp_neg_error, Fermi_amp_pos_error )
Fermi_flux				=	Fermi_data['pflx_band_ergflux'].data		#	in erg.cm^{-2}.s^{-1}.
Fermi_flux_error		=	Fermi_data['pflx_band_ergflux_error'].data	#	same as above.
Fermi_Epeak				=	Fermi_data['pflx_band_epeak'].data			#	in keV.
Fermi_Epeak_pos_error	=	Fermi_data['pflx_band_epeak_pos_err'].data	#	same as above.
Fermi_Epeak_neg_error	=	Fermi_data['pflx_band_epeak_neg_err'].data	#	same as above.
Fermi_Epeak_error		=	np.maximum( Fermi_Epeak_neg_error, Fermi_Epeak_pos_error )
Fermi_alpha				=	Fermi_data['pflx_band_alpha'].data
Fermi_alpha_pos_error	=	Fermi_data['pflx_band_alpha_pos_err'].data
Fermi_alpha_neg_error	=	Fermi_data['pflx_band_alpha_neg_err'].data
Fermi_alpha_error		=	np.maximum( Fermi_alpha_neg_error, Fermi_alpha_pos_error )
Fermi_beta				=	Fermi_data['pflx_band_beta'].data
Fermi_beta_pos_error	=	Fermi_data['pflx_band_beta_pos_err'].data
Fermi_beta_neg_error	=	Fermi_data['pflx_band_beta_neg_err'].data
Fermi_beta_error		=	np.maximum( Fermi_beta_neg_error, Fermi_beta_pos_error )
Fermi_num				=	Fermi_GRB_name.size
print 'Total number of Fermi GRBs			 	: ', Fermi_num
inds					=	np.where( np.ma.getmask( Fermi_Epeak ) == False )
Fermi_GRB_name			=	Fermi_GRB_name[inds]
Fermi_Ttimes			=	Fermi_Ttimes[inds]
Fermi_RAs				=	Fermi_RAs[inds]
Fermi_Decs				=	Fermi_Decs[inds]
Fermi_error_radius		=	Fermi_error_radius[inds]
Fermi_T90				=	Fermi_T90[inds]
Fermi_T90_error			=	Fermi_T90_error[inds]
Fermi_fluence			=	Fermi_fluence[inds]
Fermi_fluence_error		=	Fermi_fluence_error[inds]
Fermi_amp				=	Fermi_amp[inds]
Fermi_amp_pos_error		=	Fermi_amp_pos_error[inds]
Fermi_amp_neg_error		=	Fermi_amp_neg_error[inds]
Fermi_amp_error			=	Fermi_amp_error[inds]
Fermi_flux				=	Fermi_flux[inds]
Fermi_flux_error		=	Fermi_flux_error[inds]
Fermi_Epeak				=	Fermi_Epeak[inds]			
Fermi_Epeak_pos_error	=	Fermi_Epeak_pos_error[inds]
Fermi_Epeak_neg_error	=	Fermi_Epeak_neg_error[inds]
Fermi_Epeak_error		=	Fermi_Epeak_error[inds]	
Fermi_alpha				=	Fermi_alpha[inds]
Fermi_alpha_pos_error	=	Fermi_alpha_pos_error[inds]
Fermi_alpha_neg_error	=	Fermi_alpha_neg_error[inds]
Fermi_alpha_error		=	Fermi_alpha_error[inds]
Fermi_beta				=	Fermi_beta[inds]
Fermi_beta_pos_error	=	Fermi_beta_pos_error[inds]
Fermi_beta_neg_error	=	Fermi_beta_neg_error[inds]
Fermi_beta_error		=	Fermi_beta_error[inds]
Fermi_num				=	Fermi_GRB_name.size
print '...subset :		       with spectral parameters : ', Fermi_num
inds_unphysical				=	np.where( (Fermi_alpha < Fermi_beta) | (np.abs(Fermi_Epeak_error/Fermi_Epeak) > 1) )[0]
Fermi_GRB_name				=	np.delete( Fermi_GRB_name            , inds_unphysical )
Fermi_Ttimes				=	np.delete( Fermi_Ttimes              , inds_unphysical )
Fermi_RAs					=	np.delete( Fermi_RAs                 , inds_unphysical )
Fermi_Decs					=	np.delete( Fermi_Decs                , inds_unphysical )
Fermi_error_radius			=	np.delete( Fermi_error_radius        , inds_unphysical )
Fermi_T90					=	np.delete( Fermi_T90                 , inds_unphysical )
Fermi_T90_error				=	np.delete( Fermi_T90_error           , inds_unphysical )
Fermi_fluence				=	np.delete( Fermi_fluence             , inds_unphysical )
Fermi_fluence_error			=	np.delete( Fermi_fluence_error       , inds_unphysical )
Fermi_amp					=	np.delete( Fermi_amp                 , inds_unphysical )
Fermi_amp_error				=	np.delete( Fermi_amp_error           , inds_unphysical )
Fermi_amp_pos_error			=	np.delete( Fermi_amp_pos_error       , inds_unphysical )
Fermi_amp_neg_error			=	np.delete( Fermi_amp_neg_error       , inds_unphysical )
Fermi_flux					=	np.delete( Fermi_flux                , inds_unphysical )
Fermi_flux_error			=	np.delete( Fermi_flux_error          , inds_unphysical )
Fermi_Epeak					=	np.delete( Fermi_Epeak               , inds_unphysical )
Fermi_Epeak_error			=	np.delete( Fermi_Epeak_error         , inds_unphysical )
Fermi_Epeak_pos_error		=	np.delete( Fermi_Epeak_pos_error     , inds_unphysical )
Fermi_Epeak_neg_error		=	np.delete( Fermi_Epeak_neg_error     , inds_unphysical )
Fermi_alpha					=	np.delete( Fermi_alpha               , inds_unphysical )
Fermi_alpha_error			=	np.delete( Fermi_alpha_error         , inds_unphysical )
Fermi_alpha_pos_error		=	np.delete( Fermi_alpha_pos_error     , inds_unphysical )
Fermi_alpha_neg_error		=	np.delete( Fermi_alpha_neg_error     , inds_unphysical )
Fermi_beta					=	np.delete( Fermi_beta                , inds_unphysical )
Fermi_beta_error			=	np.delete( Fermi_beta_error          , inds_unphysical )
Fermi_beta_pos_error		=	np.delete( Fermi_beta_pos_error      , inds_unphysical )
Fermi_beta_neg_error		=	np.delete( Fermi_beta_neg_error      , inds_unphysical )
Fermi_num					=	Fermi_GRB_name.size
print '...subset :		   physical spectral parameters : ', Fermi_num, '\n\n'

print '\n\n\n\n'
ratio_of_fluxes	=	( Fermi_fluence / Fermi_T90 ) / Fermi_flux
print np.median( ratio_of_fluxes ), mf.std_via_mad( ratio_of_fluxes )
print np.mean( ratio_of_fluxes ), np.std( ratio_of_fluxes )
print '\n\n\n\n'


Fermi_GRB_ID=	np.zeros( Fermi_num )
Fermi_Tt	=	np.zeros( Fermi_num )
Fermi_RA	=	np.zeros( Fermi_num )
Fermi_Dec	=	np.zeros( Fermi_num )
for j, Ttime in enumerate(Fermi_Ttimes):
	
	name		=	Fermi_GRB_name[j]
	ID			=	name[3:9]
	Fermi_GRB_ID[j]	=	ID
	
	hour		=	float( Ttime[11:13] )
	mins		=	float( Ttime[14:16] )
	secs		=	float( Ttime[17:23] )
	decimal		=	convert( hour, mins, secs )
	Fermi_Tt[j]	=	decimal
	
	RA			=	Fermi_RAs[j]
	hour		=	float( RA[0: 2] )
	mins		=	float( RA[3: 5] )
	secs		=	float( RA[6:10] )
	decimal		=	convert( hour, mins, secs )
	Fermi_RA[j]	=	decimal
	
	Dec			=	Fermi_Decs[j]
	sign		=	Dec[0:1]
	deg			=	float( Dec[1:3] )
	mins		=	float( Dec[4:6] )
	secs		=	float( Dec[7:9] )
	decimal		=	convert( deg , mins, secs )
	if sign == '-':	decimal = decimal * (-1)
	Fermi_Dec[j]=	decimal
	

Fermi_GRB_ID	=	Fermi_GRB_ID.astype(int)
Fermi_Tt		=	np.round( Fermi_Tt , 3 )
Fermi_RA		=	np.round( Fermi_RA , 3 )
Fermi_Dec		=	np.round( Fermi_Dec, 3 )
#~ print Fermi_Tt
#~ print Fermi_RA
#~ print Fermi_Dec

Fermi_GRBs_table	=	Table( [ Fermi_GRB_name, Fermi_GRB_ID, Fermi_Tt, Fermi_RA, Fermi_Dec, Fermi_error_radius, Fermi_T90, Fermi_T90_error, Fermi_flux, Fermi_flux_error, Fermi_Epeak, Fermi_Epeak_error, Fermi_alpha, Fermi_alpha_error, Fermi_beta, Fermi_beta_error ],
								names = [ 'Fermi name', 'Fermi ID', 'GBM Trigger-time', 'GBM RA', 'GBM Dec', 'GBM Error-radius', 'GBM T90', 'GBM T90_error', 'GBM flux', 'GBM flux_error', 'Epeak', 'Epeak_error', 'alpha', 'alpha_error', 'beta', 'beta_error' ] )
ascii.write( Fermi_GRBs_table, './../tables/Fermi_GRBs--with_spectral_parameters.txt', format = 'fixed_width', overwrite = True )


####################################################################################################################################################






####################################################################################################################################################


common_ID					=	[]
common_Swift_name			=	[]
common_Fermi_name			=	[]
common_Swift_Tt				=	[]
common_Fermi_Tt				=	[]
common_Swift_RA				=	[]
common_Fermi_RA				=	[]
common_Swift_Dec			=	[]
common_Fermi_Dec			=	[]
common_Swift_T90			=	[]
common_Fermi_T90			=	[]
common_Fermi_T90_error		=	[]
common_Swift_phoflux		=	[]
common_Swift_phoflux_error	=	[]
common_Fermi_flux			=	[]
common_Fermi_flux_error		=	[]
common_Epeak				=	[]
common_Epeak_error			=	[]
common_Epeak_pos_error		=	[]
common_Epeak_neg_error		=	[]
common_alpha				=	[]
common_alpha_error			=	[]
common_alpha_pos_error		=	[]
common_alpha_neg_error		=	[]
common_beta					=	[]
common_beta_error			=	[]
common_beta_pos_error		=	[]
common_beta_neg_error		=	[]
for j, ID in enumerate(Fermi_GRB_ID):
	ind	=	np.where( Swift_all_GRB_ID == ID )[0]
	if ind.size != 0:
		
		#~ print ind.size
		
		diff_time	=	np.abs( Fermi_Tt[j]  -  Swift_all_Tt[ind]  )
		diff_RA		=	np.abs( Fermi_RA[j]  -  Swift_all_RA[ind]  )
		diff_Dec	=	np.abs( Fermi_Dec[j] -  Swift_all_Dec[ind] )
		#~ print diff_time
		check	=	np.where( (diff_time < 10/60) & (diff_RA < 10) & (diff_Dec < 10) )[0]			#	experimentally set at the convergent value for diff_time, doesn't change beyond 5 mins, all the way up to 10 mins; similarly for RA and Dec, roughly 10 degree by 10 degree (Fermi errors).
		#~ print check
		
		#~ if ID == 100724:
			#~ print 'Here is the culprit!	' , diff_RA, diff_Dec
		
		if check.size != 0:
			#~ if check.size != 1:
				#~ print check.size
			#~ print check
			common_ind	=	ind[check][0]
			#~ print common_ind
			#~ print Fermi_RA[j], Fermi_Dec[j]
			#~ print Swift_all_RA[common_ind], Swift_all_Dec[common_ind]
			
			common_ID.append( ID )
			common_Swift_name.append(          Swift_all_GRB_name[common_ind]      )   
			common_Fermi_name.append(          Fermi_GRB_name[j]                   )
			common_Swift_Tt.append(            Swift_all_Tt[common_ind]            )   
			common_Fermi_Tt.append(            Fermi_Tt[j]                         )
			common_Swift_RA.append(            Swift_all_RA[common_ind]            )   
			common_Fermi_RA.append(            Fermi_RA[j]                         )
			common_Swift_Dec.append(           Swift_all_Dec[common_ind]           )   
			common_Fermi_Dec.append(           Fermi_Dec[j]                        )
			common_Swift_T90.append(           Swift_all_T90[common_ind]           )
			common_Fermi_T90.append(           Fermi_T90[j]                        )
			common_Fermi_T90_error.append(     Fermi_T90_error[j]                  )
			common_Swift_phoflux.append(       Swift_all_phoflux[common_ind]       )
			common_Swift_phoflux_error.append( Swift_all_phoflux_error[common_ind] )
			common_Fermi_flux.append(          Fermi_flux[j]                       )
			common_Fermi_flux_error.append(    Fermi_flux_error[j]                 )
			common_Epeak.append(               Fermi_Epeak[j]                      )
			common_Epeak_error.append(         Fermi_Epeak_error[j]                )
			common_Epeak_pos_error.append(     Fermi_Epeak_pos_error[j]            )
			common_Epeak_neg_error.append(     Fermi_Epeak_neg_error[j]            )
			common_alpha.append(               Fermi_alpha[j]                      )
			common_alpha_error.append(         Fermi_alpha_error[j]                )
			common_alpha_pos_error.append(     Fermi_alpha_pos_error[j]            )
			common_alpha_neg_error.append(     Fermi_alpha_neg_error[j]            )
			common_beta.append(                Fermi_beta[j]                       )
			common_beta_error.append(          Fermi_beta_error[j]                 )
			common_beta_pos_error.append(      Fermi_beta_pos_error[j]             )
			common_beta_neg_error.append(      Fermi_beta_neg_error[j]             )
		
		
		
		
common_ID					=	np.array( common_ID                   )
common_Swift_name			=	np.array( common_Swift_name           )
common_Fermi_name			=	np.array( common_Fermi_name           )
common_Swift_Tt				=	np.array( common_Swift_Tt             )
common_Fermi_Tt				=	np.array( common_Fermi_Tt             )
common_Swift_RA				=	np.array( common_Swift_RA             )
common_Fermi_RA				=	np.array( common_Fermi_RA             )
common_Swift_Dec			=	np.array( common_Swift_Dec            )
common_Fermi_Dec			=	np.array( common_Fermi_Dec            )
common_Swift_T90			=	np.array( common_Swift_T90            )
common_Fermi_T90			=	np.array( common_Fermi_T90            )
common_Fermi_T90_error		=	np.array( common_Fermi_T90_error      )
common_Swift_phoflux		=	np.array( common_Swift_phoflux        )
common_Swift_phoflux_error	=	np.array( common_Swift_phoflux_error  )
common_Fermi_flux			=	np.array( common_Fermi_flux           )
common_Fermi_flux_error		=	np.array( common_Fermi_flux_error     )
common_Epeak				=	np.array( common_Epeak                )
common_Epeak_error			=	np.array( common_Epeak_error          )
common_Epeak_pos_error		=	np.array( common_Epeak_pos_error      )
common_Epeak_neg_error		=	np.array( common_Epeak_neg_error      )
common_alpha				=	np.array( common_alpha                )
common_alpha_error			=	np.array( common_alpha_error          )
common_alpha_pos_error		=	np.array( common_alpha_pos_error      )
common_alpha_neg_error		=	np.array( common_alpha_neg_error      )
common_beta					=	np.array( common_beta                 )
common_beta_error			=	np.array( common_beta_error           )
common_beta_pos_error		=	np.array( common_beta_pos_error       )
common_beta_neg_error		=	np.array( common_beta_neg_error       )
common_num					=	common_ID.size
print 'Culprit GRB100724 is still present in the common sample : ', ( common_ID == 100724 ).any()
print 'Total number of GRBs common in Swift and Fermi		: ', common_num

#~ print '\n\n\n\n'
#~ ratio_of_fluxes	=	( common_Swift_fluence / common_Swift_T90 ) / common_Fermi_flux
#~ print np.median( ratio_of_fluxes ), mf.std_via_mad( ratio_of_fluxes )
#~ print np.mean( ratio_of_fluxes ), np.std( ratio_of_fluxes )
#~ print '\n\n\n\n'


ax = plt.subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel( r'$ Swift \; \, \rm{ T_{90} \; [sec] } $', fontsize = size_font )
ax.set_ylabel( r'$ Fermi \; \, \rm{ T_{90} \; [sec] } $', fontsize = size_font )
ax.errorbar( common_Swift_T90, common_Fermi_T90, yerr =  common_Fermi_T90_error, fmt = '.', ms = marker_size, color = 'silver', markerfacecolor = 'k', markeredgecolor = 'k' )
plt.savefig( './../plots/comparing_T90s_of_common_GRBS--all.png' )
plt.clf()
plt.close()

T90_cut = 2 # in sec.
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
print 'short in Fermi only	:	', ind_short_Fermi_only.size, ind_short_Fermi_only, common_ID[ind_short_Fermi_only], common_Fermi_name[ind_short_Fermi_only], common_Swift_name[ind_short_Fermi_only], common_Fermi_T90[ind_short_Fermi_only], common_Fermi_T90_error[ind_short_Fermi_only], common_Swift_T90[ind_short_Fermi_only]
print 'short in Swift only	:	', ind_short_Swift_only.size, ind_short_Swift_only, common_ID[ind_short_Swift_only], common_Fermi_name[ind_short_Swift_only], common_Swift_name[ind_short_Swift_only], common_Fermi_T90[ind_short_Swift_only], common_Fermi_T90_error[ind_short_Swift_only], common_Swift_T90[ind_short_Swift_only]
print '\n\n'



common_GRBs_table	=	Table( [ common_ID, common_Swift_name, common_Fermi_name, common_Swift_Tt, common_Fermi_Tt, common_Swift_RA, common_Fermi_RA, common_Swift_Dec, common_Fermi_Dec,
								 common_Swift_T90, common_Fermi_T90, common_Fermi_T90_error, common_Fermi_flux, common_Fermi_flux_error,
								 common_Epeak, common_Epeak_error, common_alpha, common_alpha_error, common_beta, common_beta_error ], 
								 names = [ 'common ID', 'Swift name', 'Fermi name', 'BAT Trigger-time', 'GBM Trigger-time', 'BAT RA', 'GBM RA', 'BAT Dec', 'GBM Dec', 
											'BAT T90', 'GBM T90', 'GBM T90_error', 'GBM flux', 'GBM flux_error',
											'Epeak', 'Epeak_error', 'alpha', 'alpha_error', 'beta', 'beta_error' ]                                     )
ascii.write( common_GRBs_table, './../tables/common_GRBs--all.txt', format = 'fixed_width', overwrite = True )
