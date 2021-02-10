# xrtpy


Getting Started
===============

XRTpy is a Python Package for analyzing data from the X-Ray Telescope (XRT)
instrument onboard the Hinode spacecraft. XRTpy creation began early 2021. 
XRTpy motivation is allow users to freely use XRT data using Python. 
XRTpy will make use of SunPy, an open-source Python library for Solar
Physics data analysis and visualization.


About XRT
---------

X-Ray Telescope (XRT) is an instrument on board the Hinode spacescaft. The
XRT takes high-resolution soft X-ray images of the solar corona. XRT uses
two filter wheels, FilterWheel-1 and FilterWheel-2, both are located in 
front the CCD. Each wheels hold 6 filters. 

- FW-1 Open,Thin-Al/Poly, C/Poly, Thin-Be, Med-Be, and Med-Al.
 
- FW-2 Open, Thin-Al/Mesh, Ti/Poly, G-band, Thick-Al, and  Thick-Be. 

The XRT provides 2 arc second resolution images of the highest
temperature solar coronal material, from 1,000,000 to 10,000,000 Kelvin.
This features of XRT allows for a while temperature coverage to see all 
the coronal features. 


Samples 
-------
Sample section contains examples on how to use xrtpy to compute XRT data


### Calculating Wavelength Response Function


The effective area is computed for a set of XRT x-ray channels [units [cm^2]
as a function of wavelength [A] ]. This can be computed for ever x-ray channel.  
XRT hold 9 different channels which convert to 15 different x-ray channel combination. 


Referencing Narukage et al. (2011), the effective area [A eff] is giving by 

	A_{eff}=A \times T_{pf} \times R_{M1} \time R_{M2} \time T_{FPAF1} \times T_{FPAF2}  \times QE_{CCD}

where: 
- A Aperture
- T PF transmission of the pre-filter
- RM1, RM2 selectivities at the primary and secondary mirrors
- T FPAF1, T FPAF2 transmission of the focal-plane filters 
- QE CCD quantum efficiency of the CCD


The effective area can be calculation for all channels. 




Plan Functionality
------------------

xrtpy response
 
- xrtpy wave response 

	- Produce the effective areas and spectral responses for a set of 
	  XRT x-ray channels, accounting for some thickness of the CCD
	  contamination layer. 
	
- xrtpy temperature response
 
	- Produce the temperature response for each XRT x-ray channel,
	  assuming a spectral emission model. 


