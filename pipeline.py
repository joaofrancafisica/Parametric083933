import sys
import autolens as al
import autofit as af
import numpy as np
from lenstronomy.Data.psf import PSF

bandpass = str(sys.argv[1])

# physical useful constants and dictionarys
seeingfwhm_dict = {
    'G':0.85,
    'I':0.61,
    'R':np.nan,
    'Z':np.nan,
    'Y':np.nan
}
if bandpass not in seeingfwhm_dict.keys():
    raise Exception('Given bandpass not in bandpass dictionary.')

# physical useful constants
results_pre_path = './Results/'
exposure_time = 285 # exposure map following reference https://arxiv.org/pdf/2002.01611.pdf
pixel_scales = 20/120 #arcsec
zl = 0.26996
zs = 0.61

# residual image
residual_image = al.Array2D.from_fits(file_path=results_pre_path+'GalfitSub.fits', pixel_scales=pixel_scales, hdu=3)
# noise map
noise_map = al.Array2D.from_fits(file_path=results_pre_path+'LensLightSubNoiseMap01.fits', pixel_scales=pixel_scales, hdu=0)
# psf 
#psf = al.Kernel2D.from_fits(file_path=results_pre_path+'SExtractorPSF.fits', pixel_scales=pixel_scales, hdu=0, normalize=True)

mean_seeing_survey = seeingfwhm_dict['I']

psf_class = PSF(
    psf_type='GAUSSIAN',
    fwhm=mean_seeing_survey,
    pixel_size=pixel_scales,
    truncation=3/mean_seeing_survey
)
psf = 10*psf_class.kernel_point_source/np.max(psf_class.kernel_point_source)
psf=al.Kernel2D.manual(psf, pixel_scales=pixel_scales, shape_native=psf.shape)

# imaging object
imaging = al.Imaging(image=residual_image, noise_map=noise_map, psf=psf)

# defining our mask
circ_mask = al.Mask2D.circular(shape_native=(120, 120), pixel_scales=pixel_scales, radius=5.0)
# reading an external mask
#mask = al.Mask2D.from_fits(file_path=results_pre_path+'SExtractorMask.fits', pixel_scales=pixel_scales)
'''
combined_masks = mask + circ_mask
for m in range(0, len(combined_masks)):
    for n in range(0,len(combined_masks[m])):
        if combined_masks[m][n]>1:
            combined_masks[m][n] = 1
'''
# applying our mask
imaging = imaging.apply_mask(mask=circ_mask)

# Step 0 - source parametric (source: Sersic, lens: SIE + shear)

# We define a bright object called bulge and combine it with our galaxy model. Here, we used a light sersic profile. If you want to read more about it, please visit https://en.wikipedia.org/wiki/Sérsic_profile
bulge_0 = af.Model(al.lmp.EllSersic)
source_galaxy_model_0 = af.Model(al.Galaxy,
                                 redshift=zs,
                                 bulge=bulge_0)

mass_0 = af.Model(al.mp.EllIsothermal)
shear_0 = af.Model(al.mp.ExternalShear)
lens_galaxy_model_0 = af.Model(al.Galaxy,
                               redshift=zl,
                               mass=mass_0,
                               shear=shear_0)  

#lens_galaxy_model_0.mass.elliptical_comps.elliptical_comps_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
#lens_galaxy_model_0.mass.elliptical_comps.elliptical_comps_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
# Combining our galaxys into a single object
model_0 = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model_0, source=source_galaxy_model_0))

# Lets start to fit data and model!
search_0 = af.DynestyStatic(path_prefix = './', # Prefix path of our results
                            name = 'SourceParametric0', # Name of the dataset
                            unique_tag = 'Z_'+bandpass, # File path of our results
                            nlive = 250, # Number of live points of our dynesty sample
                            dlogz = 0.05,
                            number_of_cores = 4) # Be carefull here! This corresponds to how much core of your cpu you are going to use!
# analysis and results

analysis_0 = al.AnalysisImaging(dataset=imaging) # Passing our data through the search
result_0 = search_0.fit(model=model_0, analysis=analysis_0) # Finally fit our model

model_image = result_0.unmasked_model_image
model_image.output_to_fits(results_pre_path+'SourceParametricModelZ.fits', overwrite=True)