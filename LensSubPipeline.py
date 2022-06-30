import sys
import numpy as np
from astropy.io import fits
from lenstronomy.Data.psf import PSF
import sep
import autolens as al
import autofit as af

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
cutouts_20asec_path = './Data/20asec/'
export_pre_path = './Results/'
exposure_time = 285 # exposure map following reference https://arxiv.org/pdf/2002.01611.pdf
pixel_scales = 20/120 #arcsec
zl = 0.26996
zs = 0.61
# Step 0 - First lens light modelization (source: -, lens: Sérsic)

# reading the HSC image and generating a noise map
image_I = fits.open(cutouts_20asec_path+'J083933.4-014044.4-HSC-'+bandpass+'-pdr3_wide.fits')[1].data

#image_G = fits.open(cutouts_20asec_path+'J083933.4-014044.4-HSC-G-pdr3_wide.fits')[1].data

image = image_I

HDU_image = fits.PrimaryHDU(data=image)
HDU_image.header['EXPTIME'] = exposure_time
HDU_image.writeto(export_pre_path+'LensLightSubImage01.fits', overwrite=True)

noise_map = np.sqrt((image*exposure_time + np.var(image*exposure_time)))/exposure_time

HDU_noise_map = fits.PrimaryHDU(data=noise_map)
HDU_noise_map.writeto(export_pre_path+'LensLightSubNoiseMap01.fits', overwrite=True)

# generate a PSF
'''
psf_class = PSF(
    psf_type='GAUSSIAN',
    fwhm=seeingfwhm_dict['ze=pixel_scales
)
psf = psf_class.kernel_point_source/np.max(psf_class.kernel_point_source)
HDU_psf = fits.PrimaryHDU(data=psf)
HDU_psf.writeto(export_pre_path+'LensLightSubPSF01.fits', overwrite=True)
'''
# generate a mask
# get a mask with SExtractor (sep python)
image = image.byteswap().newbyteorder()

bkg = sep.Background(image)
data_sub = image - bkg

# objects list and a segmentation image
objs, segimage = sep.extract(
    data=data_sub,
    thresh=2.,
    minarea=20,
    deblend_nthresh=100,
    deblend_cont=0.0000005,
    err=bkg.globalrms,
    segmentation_map=True
)

HDU_segmentation = fits.PrimaryHDU(data=segimage)
HDU_segmentation.writeto(export_pre_path+'LensLightSubSegmentation01.fits', overwrite=True)

central_value = segimage[58][58]
mask = np.ones(shape=segimage.shape)
for m in range(0, len(segimage)):
    for n in range(0, len(segimage[m])):
        if segimage[m][n] == central_value or segimage[m][n] == 0:
            mask[m][n] = 0

HDU_mask = fits.PrimaryHDU(data=mask)
HDU_mask.writeto(export_pre_path+'LensLightSubMask01.fits', overwrite=True)

image = al.Array2D.from_fits(file_path=export_pre_path+'LensLightSubImage01.fits', pixel_scales=pixel_scales, hdu=0)
noise_map = al.Array2D.from_fits(file_path=export_pre_path+'LensLightSubNoiseMap01.fits', pixel_scales=pixel_scales, hdu=0)
psf = al.Kernel2D.from_fits(file_path=export_pre_path+'SExtractorPSF.fits', pixel_scales=pixel_scales, hdu=0, normalize=True)
mask = al.Mask2D.from_fits(file_path=export_pre_path+'LensLightSubMask01.fits', pixel_scales=pixel_scales)

# stablishing PyAutoLens objects
circ_mask = al.Mask2D.circular(shape_native=(120, 120), pixel_scales=pixel_scales, radius=5.)
# comining masks
combined_masks = mask + circ_mask
for m in range(0, len(combined_masks)):
    for n in range(0,len(combined_masks[m])):
        if combined_masks[m][n] > 1:
            combined_masks[m][n] = 1

# imaging object
imaging = al.Imaging(image=image, noise_map=noise_map, psf=psf)
imaging = imaging.apply_mask(mask=combined_masks)

# source galaxy model
source_galaxy_model = af.Model(al.Galaxy, redshift=zs)
# lens galaxy model
lens_galaxy_model = af.Model(al.Galaxy, redshift=zl, bulge_0=al.lmp.EllSersic)
# model object
lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))

# search object
search = af.DynestyStatic(path_prefix='./',
                          name = 'LensLightSub01',
                          unique_tag = 'PSF'+bandpass,
                          nlive = 50,
                          number_of_cores = 4) # be carefull here! verify your core numbers
# analysis object
analysis = al.AnalysisImaging(dataset=imaging)
# results object
result_0 = search.fit(model=lens_light_model, analysis=analysis)

# Step 1 - First lens light modelization (source: -, lens: Sérsic+EXP)
result_0_unmaskedmodel = result_0.unmasked_model_image

lenslightsub01 = image - result_0_unmaskedmodel
lenslightsub01.output_to_fits(file_path=export_pre_path+'LensLightSub01.fits', overwrite=True)
lenslightsub01 = fits.open(export_pre_path+'LensLightSub01.fits')[0].data

lenslightsub01 = lenslightsub01.byteswap().newbyteorder()

bkg = sep.Background(lenslightsub01)
data_sub = lenslightsub01 - bkg

# objects list and a segmentation image
objs, segimage = sep.extract(
    data=data_sub,
    thresh=1.5,
    minarea=20,
    deblend_nthresh=100,
    deblend_cont=0.0000005,
    err=bkg.globalrms,
    segmentation_map=True
)

HDU_segmentation = fits.PrimaryHDU(data=segimage)
HDU_segmentation.writeto(export_pre_path+'LensLightSubSegmentation02.fits', overwrite=True)

central_value = segimage[58][58]
mask = np.ones(shape=segimage.shape)
for m in range(0, len(segimage)):
    for n in range(0, len(segimage[m])):
        if segimage[m][n] == central_value or segimage[m][n] == 0:
            mask[m][n] = 0

HDU_mask = fits.PrimaryHDU(data=mask)
HDU_mask.writeto(export_pre_path+'LensLightSubMask02.fits', overwrite=True)

mask = al.Mask2D.from_fits(file_path=export_pre_path+'LensLightSubMask02.fits', pixel_scales=pixel_scales)

# comining masks
combined_masks = mask + circ_mask
for m in range(0, len(combined_masks)):
    for n in range(0,len(combined_masks[m])):
        if combined_masks[m][n]>1:
            combined_masks[m][n] = 1

# imaging object
imaging = al.Imaging(image=image, noise_map=noise_map, psf=psf)
imaging = imaging.apply_mask(mask=combined_masks)

# source galaxy model
source_galaxy_model = result_0.model.galaxies.source # same as before
# lens galaxy model
lens_galaxy_model = af.Model(al.Galaxy, redshift=zl, bulge_0=al.lmp.EllSersic, bulge_1=al.lmp.EllExponential)
# model object
lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))

# search object
search = af.DynestyStatic(path_prefix='./',
                          name = 'LensLightSub02',
                          unique_tag = 'PSF'+bandpass,
                          nlive = 350,
                          number_of_cores = 4, # be carefull here! verify your core numbers
                          dlogz=0.05) 
# analysis object
analysis = al.AnalysisImaging(dataset=imaging)
# results object
result_1 = search.fit(model=lens_light_model, analysis=analysis)

result_1_unmaskedmodel = result_1.unmasked_model_image

lenslightsub02 = image - result_1_unmaskedmodel
lenslightsub02.output_to_fits(file_path=export_pre_path+'LensLightSub02.fits', overwrite=True)
lenslightsub02 = fits.open(export_pre_path+'LensLightSub02.fits')[0].data