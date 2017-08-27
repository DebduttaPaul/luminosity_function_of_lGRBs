"""

This python script contains a list of python functions that are useful in data analysis.
Each function has explanations in the beginning to understand the basic functionality.

Some functions are used in carrying out general operations in others, hence all have been included here.

Instructions to use this script
--------------------------------
This script needs to be kept in the same directory in which the main python script is being executed, and then imported in the header, thus:
"import debduttaS_functions". Alternatively, a simpler name may be adopted during the import, e.g. "df" via
"import debduttaS_functions as df". Then all the functions can be used with the module name "df", e.g.
if the function "window" is to be used, it can be called thus: "df.window(low, high, primary, secondary)".



Author:			Debdutta Paul
Last updated:	14th November, 2016

"""


from __future__ import division
import os, warnings, time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
from scipy.misc import factorial as fact
from scipy import interpolate
from scipy.optimize import curve_fit
plt.rc('axes', linewidth=2)
plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

P = np.pi # Dear old pi!
padding	 = 8 # The padding of the axes labels.
size_font = 18 # The fontsize in the images.



#######################################################################################################################################################






#######################################################################################################################################################




def window( low, high, primary, secondary ):
	
	
	'''
	
	
	Parameters
	-----------
	low:		Lower limit.
	high:		Upper limit.
	primary:	Array on which the limits are being set.
	secondary:	Another array, which has same length as "primary" and its corresponding data.
				If this array is not required, an empty array can be supplied.
	
	Returns
	-----------
	chopped_primary:	The elements in the "primary" within and including the lower and upper limits.
	chopped_secondary:	The corresponding elements in the "secondary"; or empty, if an empty array was supplied.
	
	Comments
	-----------
	Assumes that "primary" is arranged in ascending order.
	
	
	'''
	
	
	
	lower = np.where( primary >= low  )[0][0]
	upper = np.where( primary <= high )[0][-1] + 1
	
	chopped_primary = primary[lower:upper]
	
	if len(secondary)==0:	chopped_secondary = []
	else:	chopped_secondary = secondary[lower:upper]
	
	
	return	chopped_primary, chopped_secondary
	



def delete( bigger, smaller ):
	
	
	'''
	
	
	Parameters
	-----------
	bigger:		An array of length greater than that of 'smaller'.
	smaller:	An array containing a set of elements which exactly equal a subset of 'bigger'.
	
	Returns
	-----------
	deleted:	The overlying region removed from "bigger", i.e. "bigger" - "smaller" .
	
	Assumptions
	-----------
	The data in the overlying region are exactly the same.
	
	
	'''
	
	
	index = []
	
	for i in range( len(smaller) ):
		index.append( np.where(bigger == smaller[i])[0][0] )
	
	deleted = np.delete(bigger, index)
	
	
	return deleted




def in_array( array, value, margin ):
	
	
	'''
	
	
	Parameters 
	-----------
	array:		An array in which "value" is to be searched for.
	value:		The float variable that is to be looked for in "array" .
	margin:		The accuracy to which the match has to be be looked for.
	
	Returns
	-----------
	B:			Boolean variable that is 1 if "value" was found in "array" and 0 if it was not.
	i:			The first index of "array" in which "value" was found if B = 1; garbage large number (10^20) if B = 0.
	
	
	'''
	
	
	
	B	=	0
	l	=	len(array)
	
	ind = []
	
	for j in range(l):
		
		if np.abs( array[j] - value ) < margin:
			B = 1
			ind.append(j)
	
	if len(ind) > 0:
		i = ind[0]
	else:
		i = 1e20
	
	
	return B, i
	



def nearest( array, value ):
	
	
	'''
	
	
	Parameters
	-----------
	array:	An array, which is to be searched over.
	value:	The value that is being searched in "array".
	
	Returns
	-----------
	index:	The index in "array" corresponding to which element is the closest to "value".
	
	
	Assumptions
	-----------
	"value" lies close to at least one element in "array".
	
	
	'''
	
	
	diff	=	np.abs( array - value )
	minimum	=	np.min( diff )
	index	=	np.where( diff == minimum )[0][0]
	
	return index




def std_via_mad( array ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		Array of values of which STD (standard deviation) is to be calculated via MAD (median absolute deviation).
	
	Returns
	-----------
	std:		Robust standard deviation of "array".
	
	
	'''
	
	
	
	med		=	np.median(array)
	mad	=	np.median( np.abs(array - med) )
	std	=	1.4826 * mad
	
	return std




def rebin( array, intervals ):
		
	
	'''
	
	
	Parameters
	-----------
	array:		The array which needs to divided into equal "intervals".
	intervals:	The number of intervals required.
	
	Returns
	-----------
	limits:		Array of arrays, each containing upper and lower limits of the rebinned array.
	
	
	'''
	
	
	
	array_lo = np.min(array)
	array_hi = np.max(array)
	
	A = np.linspace(array_lo, array_hi, intervals + 1 )
	
	limits = []
	
	for i in range( 0, intervals ):
		limits.append(  np.array( [ A[i], A[i+1] ] )  )
	
	limits = np.array(limits)
	
	
	return limits
	



def my_range( array ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		The array whose range is to be found.
	
	Returns
	-----------
	range_of_array:		The required range.
	
	
	'''
	
	
	
	range_of_array = np.max(array) - np.min(array)	
	
	return range_of_array




def mkdir_p( my_path ):
	
	
    '''
	
	
	Parameters
	-----------
	my_path:	The path in which the directory is to be made.
	
	Comments
    -----------
	Creates a directory (equivalent to using "mkdir -p" on the command line).
	
	
	'''
	
	
    from errno import EEXIST
    from os import makedirs, path
	
    try:
        makedirs(my_path)
    except OSError as exc: # Python > 2.5
        
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise
        return




def poisson( k, mean, coeff ):
	
	
	'''
	
	
	Parameters 
	----------
	k:				The real number which is to be mapped to a poisson distribution function (pdf).
	mean:			The lambda parameter (mean) of the pdf.
	coeff:			The coefficient of the pdf.
	
	Returns
	----------
	poiss_k:		The mapped real number.
	
	
	'''
	
	
	poiss_k = coeff * ( mean**k / fact(k) )
	
	return poiss_k




def gaussian( x, mu, sigma, coeff ):
	
	
	'''
	
	
	Parameters 
	----------
	x:			The real number which is to be mapped to a gaussian distribution (gd).
	mu:			The mean of the gd.
	sigma:		The standard deviation of the gd.
	coeff:		The coefficient of the gd.
	
	Returns
	----------
	gauss_k:	The mapped real number.
	
	
	'''
	
	
	gauss_x	=	coeff * np.exp( -0.5 * ( (x-mu)/sigma )**2 )
	
	return gauss_x




def fourier_transform( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	Array consisting of data along the x-axis.
	y:	Array consisting of data along the y-axis.
	
	Returns
	-----------
	nu:	The reciprocal space of "x".
	c:	The corresponding fourier co-efficients (complex numbers).
		
	Caveats
	-----------
	For the case when "x" is non-uniformly gridded, the Fourier transform may not be accurate. A model dataset is created with uniform binning, the bin-size being the mean of the intervals. 
	
	
	
	'''
	
	
	step	= np.mean( x[1:] - x[:-1] )
	
	c_raw	= np.fft.fft(y)
	c		= np.fft.fftshift(c_raw)
	
	nu_raw	=	np.fft.fftfreq( len(x), d = step )
	nu		=	np.fft.fftshift( nu_raw )
	
	l = len(nu)/2 + 1
	
	nu		=	nu[l:]
	c		=	 c[l:]
	
	
	return nu, c




def mid_array( input_array ):
	
	
	'''
	
	
	Parameters
	-----------
	input_array:	The array for which the midpoint between consecutive elements will be delivered.
	
	Returns
	-----------
	output_array:	The required array.
	
	Comments
	-----------
	This function is to provide the array of midpoints for the cases when the np.histogram returns 2 arrays of unequal length
	The length of edges-array is 1 greater than the values-array, so to match the values with the edge this function is used.
	
	
	'''
	
	
	output_array = ( input_array[:-1] + input_array[1:] ) / 2
	
	
	return output_array




def distance( p1, p2 ):
	
	
	'''
	
	
	Parameters
	----------
	p1:	An array or list containing the x & y co-ordinates, respectively, of the first point on a plane.
	p2:	Similarly for the second point.
	
	Returns
	---------
	d: The distance between the two points.
	
	
	'''
	
	
	d = ( p1[0] - p2[0] )**2 + ( p1[1] - p2[1] )**2
	d = np.sqrt(d)
	
	return d




def radian_to_degree( angle_in_radian ):
	
	
	'''
	
	
	Parameters
	-----------
	angle_in_radian:	The real number to be converted into degree.	
	
	Returns
	-----------
	angle_in_degree:	The same in degree.
	
	
	'''
	
	
	import numpy as np
	P	=	np.pi
	
	angle_in_degree	=	angle_in_radian * 180/P
	
	return angle_in_degree




def degree_to_radian (angle_in_degree ):
	
	
	'''
	
	
	Parameters
	-----------
	angle_in_degree:	The angle, in degree, to be converted into a real number.
	
	Returns
	-----------
	angle_in_radian:	The required real number.
	
	
	'''
	
	
	import numpy as np
	P	=	np.pi
	
	angle_in_radian	=	angle_in_degree * P/180
	
	return angle_in_radian




def my_histogram_according_to_given_boundaries( array, bin_size, low, high ):
	
	
	'''
	
	
	Parameters
	-----------
	array	:	Array, which contains only integer values, which is to be histogram-ed according to these histograms.
	bin_size:	The size of the bin for making the histogram.
	low		:	The lower boundary of the histogram.
	high	:	The upper boundary of the histogram.
	
	Returns
	-----------
	x:		The x-axis of the required histogram.
	y:		The y-axis of the required histogram.
	
	
	'''
	
	
	array	=	np.sort( array )
	bin_edges	=	np.arange( low, high+bin_size/2, bin_size )
	x	=	mid_array( bin_edges )
	
	#	print array, bin_edges, x
	
	hist	=	np.histogram( array, bin_edges )
	
	y = hist[0]
	
	
	return x, y




def my_histogram_with_last_bin_removed( array, bin_size ):
	
	
	'''
	
	
	Parameters
	-----------
	array:		Array whose histogram is required.
	bin_size:	Bin size for creating the histogram, in same units as above array.
	
	Returns
	-----------
	x:			Array of rebinned elements.
	y:			Array of same length, giving the number of events in the new bins.
	
	Assumes
	-----------
	The last bin has incomplete data and is to be ignored.
	
	
	'''
	
	
	array	=	np.sort( array )
	a_range	=	my_range( array )
	
	number_of_bins	=	int( a_range / bin_size )
	
	rem	=	a_range % bin_size
	
	if rem != 0:
		stop 		=	number_of_bins*bin_size + array[0]
		array, []	=	window( array[0], stop, array, [] )
		array[-1]	=	stop
	
	bin_edges	=	np.arange( array[0], array[-1]+bin_size, bin_size )
	x	=	mid_array( bin_edges )
				
	hist	=	np.histogram( array, bin_edges )
	
	y = hist[0]
	
	
	return x, y




def reduced_chisquared( theoretical, observed, obs_err, constraints ):
	
	
	'''
	
	Parameters
	-----------
	theoretical:		Array of theoretical quantity.
	observed:			Array of observed quantity.
	obs_err:			Array of errors on observed quantity.
	constraints:		The number of constraints on the fitting.
	
	Returns
	-----------
	chisqrd:			The chisqrd.
	dof:				The number of degrees of freedom.
	reduced_chisquared:	The reduced chi-squared.
	
		
	'''
	
		
	chisqrd	=	np.sum(   ( (observed-theoretical)/obs_err )**2   )
	dof		=	len(observed)-constraints
	reduced_chisquared	=	chisqrd / dof
		
	return chisqrd, dof, reduced_chisquared




def sort( primary, secondary ):
	
	
	'''
	
	
	Parameters
	-----------
	primary:	The array which is to be sorted.
	secondary:	Another array which is to be sorted according to the elements of 'primary'.
	
	Returns
	----------
	p:		Sorted primary.
	s:		Correspondingly sorted secondary.
	
	
	'''
	
	
	index	=	np.argsort( primary )
	p	=	primary[index]
	s	=	secondary[index]
	
	
	return p, s




def fit_a_poisson( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	The array containing the independent discrete variable.
	y:	The dependent variable which is to be attempted to fit to a Poisson distribution function.
	
	Returns
	-----------
	mean_best:	Best-fit mean.
	coeff_best:	Best-fit co-efficient.
	
	Comments
	-----------
	Assumes that the fit will not fail drastically!
	
	
	'''
	
	
	masks		=	np.where( y > 0 )[0]
	x_masked	=	x[masks]
	y_masked	=	y[masks]
	
	mean0		=	np.sum( y_masked * x_masked ) / np.sum(y_masked)
	coeff0		=	np.exp( -mean0 ) * np.sum(y_masked)
	
	popt, pcov	=	curve_fit( poisson, x_masked, y_masked, p0 = [mean0, coeff0] )
	mean_best	=	popt[0]
	coeff_best	=	popt[1]
	
	
	return mean_best, coeff_best




def fit_a_gaussian( x, y ):
	
	
	'''
	
	
	Parameters
	-----------
	x:	The array containing the independent discrete variable.
	y:	The dependent variable which is to be attempted to fit to a Poisson distribution function.
	
	Returns
	-----------
	mean_best:	Best-fit mean.
	sigma_best:	Best-fit sigma.
	coeff_best:	Best-fit co-efficient.
	
	Comments
	-----------
	Assumes that the fit will not fail drastically!
	
	
	'''
	
	
	masks		=	np.where( y > 0 )[0]
	x_masked	=	x[masks]
	y_masked	=	y[masks]
	
	coeff0		=	np.max(y_masked)
	mean0		=	x_masked[ np.where(y_masked==coeff0)[0][0] ]
	sigma0		=	np.sqrt(  np.sum( y_masked * (x_masked-mean0)**2 ) / np.sum(y_masked)  )
	
	popt, pcov	=	curve_fit( gaussian, x_masked, y_masked, p0 = [mean0, sigma0, coeff0] )
	mean_best	=	popt[0]
	sigma_best	=	np.abs( popt[1] )
	coeff_best	=	popt[2]
	
	
	return mean_best, sigma_best, coeff_best




def stats_over_array( array, N ):
	
	
	'''
	
	
	Parameters
	-----------
	array:	Array which is to be binned every "N" data points.
	N:		Number of points.
	
	Returns
	-----------
	array_sum:	Over the array binned every "N" data points, sum.
	array_mean:	Over the array binned every "N" data points, mean.
	array_std:	Over the array binned every "N" data points, standard deviation.
	array_var:	Over the array binned every "N" data points, varaiance.
	
	Comments
	-----------
	After rebinning, throws away data in the last bin, if it is incomplete.
	
	
	'''
	
	
	lx	=	len( array )
	rem	=	lx % N
	
	#	To remove the last bin with incomplete data for the required binning.
	if rem != 0:
		array	=	array[:-rem]
	
	L	=	int( lx // N )
	
	array_2D	=	array.reshape( L, N )
	array_sum	=	array_2D.sum( axis = 1 )
	array_mean	=	array_2D.mean( axis = 1 )
	array_std	=	array_2D.std( axis = 1 )
	array_var	=	array_2D.var( axis = 1 )
		
	return array_sum, array_mean, array_std, array_var


