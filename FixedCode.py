#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#file_name.py
#Python 3.6
"""
Created: Wed Nov 13 13:13:36 2024
Modified: Wed Nov 13 13:13:36 2024
Author: Emmanuel Holt

Description
-----------
"""
import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits


''' LOAD IN IMAGES '''

# Bias images


hdul1 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-001-bi.fit')
bi1 = hdul1[0].data

hdul2 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-002-bi.fit')
bi2 = hdul2[0].data

hdul3 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-003-bi.fit')
bi3 = hdul3[0].data

hdul4 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-004-bi.fit')
bi4 = hdul4[0].data

hdul5 = fits.open('//Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-005-bi.fit')
bi5 = hdul5[0].data

hdul6 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-006-bi.fit')
bi6 = hdul6[0].data

hdul7 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-007-bi.fit')
bi7 = hdul7[0].data

#%% 
''' Take Master Bias '''
    # The median of the biases


# Blank image
blank = bi1*0

# Take one data point
pixel_value = bi1[0, 0]


# Test

bi_array = np.stack([bi1, bi2, bi3, bi4, bi5, bi6, bi7])


# How to take median of the 7

master_bias = np.median(bi_array, axis=0)

# Display the master bias

'''
plt.figure()
plt.imshow(master_bias, cmap='gray')
plt.colorbar()
plt.title("Master bias")
plt.show()
'''





''' Master Dark '''

# Dark images

hdul8 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-001-d.fit')
d8 = hdul8[0].data

hdul9 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-002-d.fit')
d9 = hdul9[0].data

hdul10 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-003-d.fit')
d10 = hdul10[0].data

hdul11 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-004-d.fit')
d11 = hdul11[0].data

hdul12 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-005-d.fit')
d12 = hdul12[0].data

hdul13 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-006-d.fit')
d13 = hdul13[0].data

hdul14 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/calib-007-d.fit')
d14 = hdul14[0].data


# Subtract master bias from each dark

dark_1 = d8 - master_bias
dark_2 = d9 - master_bias
dark_3 = d10 - master_bias
dark_4 = d11 - master_bias
dark_5 = d12 - master_bias
dark_6 = d13 - master_bias
dark_7 = d14 - master_bias


# Take median

dark_array = np.stack([dark_1, dark_2, dark_3, dark_4, dark_5, dark_6, dark_7])

master_dark = np.median(dark_array, axis=0)


# Display the image
'''
plt.figure()
plt.imshow(master_dark, cmap='gray')
plt.colorbar()
plt.title("Master dark")
plt.show()
'''


#%%


''' Master Flat - O filter '''


# Loading images
# Flats - O filter

hdul15 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-001-o.fit')
f15 = hdul15[0].data

hdul16 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-002-o.fit')
f16 = hdul16[0].data

hdul17 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-003-o.fit')
f17 = hdul17[0].data

#hdul18 = fits.open('/Users/maika/Downloads/calibration/flats-004-o.fit')
#f18 = hdul18[0].data
    # For some reason no flats 004 ?

hdul19 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-005-o.fit')
f19 = hdul19[0].data

hdul20 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-006-o.fit')
f20 = hdul20[0].data

hdul21 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-007-o.fit')
f21 = hdul21[0].data

hdul22 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-008-o.fit')
f22 = hdul22[0].data



# Find time relation between Darks and Flats

t_d = 300 # s

t_f_O = 20 # s

ts_O = (t_f_O) / (t_d)




# Subtract the master bias and time scaled master dark from each flat

fO_1 = f15 - master_bias - ts_O*master_dark
fO_2 = f16 - master_bias - ts_O*master_dark
fO_3 = f17 - master_bias - ts_O*master_dark
# NO image 4!!!!
fO_5 = f19 - master_bias - ts_O*master_dark
fO_6 = f20 - master_bias - ts_O*master_dark
fO_7 = f21 - master_bias - ts_O*master_dark
fO_8 = f22 - master_bias - ts_O*master_dark


# Median combine

fO_array = np.stack([fO_1, fO_2, fO_3, fO_5, fO_6, fO_7, fO_8])

median_flat_O = np.median(fO_array, axis=0)


# Get median pixel

flatO_median_pixel = np.median(median_flat_O)


# Normalize

normal_flat_O = median_flat_O / flatO_median_pixel


# Display the image
'''
plt.figure()
plt.imshow(normal_flat_O, cmap='gray')
plt.colorbar()
plt.title("Normalized master flat O")
plt.show()
'''






''' Master Flat - H filter '''


# Loading Flats - Ha filter images


hdul23 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-001-ha.fit')
f23 = hdul23[0].data

hdul24 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-002-ha.fit')
f24 = hdul24[0].data

hdul25 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-003-ha.fit')
f25 = hdul25[0].data

hdul26 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-004-ha.fit')
f26 = hdul26[0].data

hdul27 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-005-ha.fit')
f27 = hdul27[0].data

hdul28 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-006-ha.fit')
f28 = hdul28[0].data

hdul29 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-007-ha.fit')
f29 = hdul29[0].data

hdul30 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/calibration/calibration/flats-008-ha.fit')
f30 = hdul30[0].data



# Find time relation between Darks and Flats

t_d = 300 # s

t_f_H = 10 # s

ts_H = (t_f_H) / (t_d)


# Subtract the master bias and time scaled master dark from each flat


fH_1 = f23 - master_bias - ts_H*master_dark
fH_2 = f24 - master_bias - ts_H*master_dark
fH_3 = f25 - master_bias - ts_H*master_dark
fH_4 = f26 - master_bias - ts_H*master_dark
fH_5 = f27 - master_bias - ts_H*master_dark
fH_6 = f28 - master_bias - ts_H*master_dark
fH_7 = f29 - master_bias - ts_H*master_dark
fH_8 = f30 - master_bias - ts_H*master_dark

scaled_dark = ts_H*master_dark



# Median combine

fH_array = np.stack([fH_1, fH_2, fH_3, fH_4, fH_5, fH_6, fH_7, fH_8])

median_flat_H = np.median(fH_array, axis=0)


# Get median pixel

flatH_median_pixel = np.median(median_flat_H)


# Normalize

normal_flat_H = median_flat_H / flatH_median_pixel



# Display the image

'''
plt.figure()
plt.imshow(normal_flat_H, cmap='gray')
plt.colorbar()  # Optional: adds a color bar to show pixel values
plt.title("Normalized master flat H")
plt.show()
'''




''' Now using the calibration images to correct the raw image '''


''' Start with Ha images '''


# Load the images


hdul24 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-001-ha.fit')
ha24 = hdul24[0].data

hdul25 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-002-ha.fit')
ha25 = hdul25[0].data

hdul26 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-003-ha.fit')
ha26 = hdul26[0].data

hdul27 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-004-ha.fit')
ha27 = hdul27[0].data

hdul28 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-005-ha.fit')
ha28 = hdul28[0].data

hdul29 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-006-ha.fit')
ha29 = hdul29[0].data

hdul30 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-007-ha.fit')
ha30 = hdul30[0].data

hdul31 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-008-ha.fit')
ha31 = hdul31[0].data

hdul32 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-009-ha.fit')
ha32 = hdul32[0].data

hdul33 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-010-ha.fit')
ha33 = hdul33[0].data

hdul34 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-011-ha.fit')
ha34 = hdul34[0].data

hdul35 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-012-ha.fit')
ha35 = hdul35[0].data

hdul36 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-013-ha.fit')
ha36 = hdul36[0].data

hdul37 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-014-ha.fit')
ha37 = hdul37[0].data

hdul38 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-015-ha.fit')
ha38 = hdul38[0].data

hdul39 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-016-ha.fit')
ha39 = hdul39[0].data

hdul40 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-017-ha.fit')
ha40 = hdul40[0].data

hdul41 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-018-ha.fit')
ha41 = hdul41[0].data

hdul42 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-019-ha.fit')
ha42 = hdul42[0].data

hdul43 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-020-ha.fit')
ha43 = hdul43[0].data


# Subtract master bias and time scaled master dark from each image

    # Master dark already at same time scale (300s)

t_s = 300/300 # seconds


# Also divide by the normalized flat field for H

sci_ha_1 = (ha24 - master_bias - master_dark) / (normal_flat_H)


# Load in log norm to load images in log scale
from matplotlib.colors import LogNorm


# Calibrate the rest of the science images for Ha!!

sci_ha_2 = (ha25 - master_bias - master_dark) / (normal_flat_H)
sci_ha_3 = (ha26 - master_bias - master_dark) / (normal_flat_H)
sci_ha_4 = (ha27 - master_bias - master_dark) / (normal_flat_H)
sci_ha_5 = (ha28 - master_bias - master_dark) / (normal_flat_H)
sci_ha_6 = (ha29 - master_bias - master_dark) / (normal_flat_H)
sci_ha_7 = (ha30 - master_bias - master_dark) / (normal_flat_H)
sci_ha_8 = (ha31 - master_bias - master_dark) / (normal_flat_H)
sci_ha_9 = (ha32 - master_bias - master_dark) / (normal_flat_H)
sci_ha_10 = (ha33 - master_bias - master_dark) / (normal_flat_H)
sci_ha_11 = (ha34 - master_bias - master_dark) / (normal_flat_H)
sci_ha_12 = (ha35 - master_bias - master_dark) / (normal_flat_H)
sci_ha_13 = (ha36 - master_bias - master_dark) / (normal_flat_H)
sci_ha_14 = (ha37 - master_bias - master_dark) / (normal_flat_H)
sci_ha_15 = (ha38 - master_bias - master_dark) / (normal_flat_H)
sci_ha_16 = (ha39 - master_bias - master_dark) / (normal_flat_H)
sci_ha_17 = (ha40 - master_bias - master_dark) / (normal_flat_H)
sci_ha_18 = (ha41 - master_bias - master_dark) / (normal_flat_H)
sci_ha_19 = (ha42 - master_bias - master_dark) / (normal_flat_H)
sci_ha_20 = (ha43 - master_bias - master_dark) / (normal_flat_H)





''' Start with O-III images '''


# Load the OIII images

hdul44 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-001-o.fit')
o1 = hdul44[0].data

hdul45 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-002-o.fit')
o2 = hdul45[0].data

hdul46 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-003-o.fit')
o3 = hdul46[0].data

hdul47 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-004-o.fit')
o4 = hdul47[0].data

hdul48 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-005-o.fit')
o5 = hdul48[0].data

hdul49 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-006-o.fit')
o6 = hdul49[0].data

hdul50 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-007-o.fit')
o7 = hdul50[0].data

hdul51 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-008-o.fit')
o8 = hdul51[0].data

hdul52 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-009-o.fit')
o9 = hdul52[0].data

hdul53 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-010-o.fit')
o10 = hdul53[0].data

hdul54 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-011-o.fit')
o11 = hdul54[0].data

hdul55 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-012-o.fit')
o12 = hdul55[0].data

hdul56 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-013-o.fit')
o13 = hdul56[0].data

hdul57 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-014-o.fit')
o14 = hdul57[0].data

hdul58 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-015-o.fit')
o15 = hdul58[0].data

hdul59 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-016-o.fit')
o16 = hdul59[0].data

hdul60 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-017-o.fit')
o17 = hdul60[0].data

hdul61 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-018-o.fit')
o18 = hdul61[0].data

hdul62 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-019-o.fit')
o19 = hdul62[0].data

hdul63 = fits.open('/Users/emmanuelholt/Documents/NGC7635_data/science/NGC7635-020-o.fit')
o20 = hdul63[0].data


# Subtract master bias and time scaled master dark from each image

    # Master dark already at same time scale (300s)

t_s = 300/300 # seconds


# Also divide by the normalized flat field for O

sci_o_1 = (o1 - master_bias - master_dark) / (normal_flat_O)
sci_o_2 = (o2 - master_bias - master_dark) / normal_flat_O
sci_o_3 = (o3 - master_bias - master_dark) / normal_flat_O
sci_o_4 = (o4 - master_bias - master_dark) / normal_flat_O
sci_o_5 = (o5 - master_bias - master_dark) / normal_flat_O
sci_o_6 = (o6 - master_bias - master_dark) / normal_flat_O
sci_o_7 = (o7 - master_bias - master_dark) / normal_flat_O
sci_o_8 = (o8 - master_bias - master_dark) / normal_flat_O
sci_o_9 = (o9 - master_bias - master_dark) / normal_flat_O
sci_o_10 = (o10 - master_bias - master_dark) / normal_flat_O
sci_o_11 = (o11 - master_bias - master_dark) / normal_flat_O
sci_o_12 = (o12 - master_bias - master_dark) / normal_flat_O
sci_o_13 = (o13 - master_bias - master_dark) / normal_flat_O
sci_o_14 = (o14 - master_bias - master_dark) / normal_flat_O
sci_o_15 = (o15 - master_bias - master_dark) / normal_flat_O
sci_o_16 = (o16 - master_bias - master_dark) / normal_flat_O
sci_o_17 = (o17 - master_bias - master_dark) / normal_flat_O
sci_o_18 = (o18 - master_bias - master_dark) / normal_flat_O
sci_o_19 = (o19 - master_bias - master_dark) / normal_flat_O
sci_o_20 = (o20 - master_bias - master_dark) / normal_flat_O


# Plot the first image in log scale


'''

sci_o_1_log = np.log(sci_o_1)
plt.figure()
plt.imshow(sci_o_1_log, cmap='gray',)
plt.colorbar()
plt.title("O log")
plt.show()

plt.figure()
plt.imshow(np.log(sci_o_19), cmap='gray')
plt.colorbar()
plt.title("19")
plt.show()
'''




''' Co-adding the images '''


''' Hydrogen filter '''


sci_ha_1 = (ha24 - master_bias - master_dark) / (normal_flat_H)
sci_ha_2 = (ha25 - master_bias - master_dark) / (normal_flat_H)
sci_ha_3 = (ha26 - master_bias - master_dark) / (normal_flat_H)
sci_ha_4 = (ha27 - master_bias - master_dark) / (normal_flat_H)
sci_ha_5 = (ha28 - master_bias - master_dark) / (normal_flat_H)
sci_ha_6 = (ha29 - master_bias - master_dark) / (normal_flat_H)
sci_ha_7 = (ha30 - master_bias - master_dark) / (normal_flat_H)
sci_ha_8 = (ha31 - master_bias - master_dark) / (normal_flat_H)
sci_ha_9 = (ha32 - master_bias - master_dark) / (normal_flat_H)
sci_ha_10 = (ha33 - master_bias - master_dark) / (normal_flat_H)
sci_ha_11 = (ha34 - master_bias - master_dark) / (normal_flat_H)
sci_ha_12 = (ha35 - master_bias - master_dark) / (normal_flat_H)
sci_ha_13 = (ha36 - master_bias - master_dark) / (normal_flat_H)
sci_ha_14 = (ha37 - master_bias - master_dark) / (normal_flat_H)
sci_ha_15 = (ha38 - master_bias - master_dark) / (normal_flat_H)
sci_ha_16 = (ha39 - master_bias - master_dark) / (normal_flat_H)
sci_ha_17 = (ha40 - master_bias - master_dark) / (normal_flat_H)
sci_ha_18 = (ha41 - master_bias - master_dark) / (normal_flat_H)
sci_ha_19 = (ha42 - master_bias - master_dark) / (normal_flat_H)
sci_ha_20 = (ha43 - master_bias - master_dark) / (normal_flat_H)




#%% Aligned Ha Images

# New Method:
   
    # Identify the (y, x) coordinates of the maximum pixel in each image.
   
    # Calculate the shift needed to move this maximum to a reference location (for example, the maximum location in the first image).
   
    # Apply the shift to each image so that the maximum value in each aligns with the reference location.


# Find max values

from scipy.ndimage import shift

# Assuming `images` is a list of numpy arrays representing each image
images = [sci_ha_1, sci_ha_2, sci_ha_3, sci_ha_4, sci_ha_5, sci_ha_6, sci_ha_7,
          sci_ha_8, sci_ha_9, sci_ha_10, sci_ha_11, sci_ha_12, sci_ha_13, sci_ha_14,
          sci_ha_15, sci_ha_16, sci_ha_17, sci_ha_18, sci_ha_19, sci_ha_20]

o_images = [sci_o_1,sci_o_2,sci_o_3,sci_o_4,sci_o_5,sci_o_6,sci_o_7,sci_o_8,sci_o_9,sci_o_10,
            sci_o_11,sci_o_12,sci_o_13,sci_o_14,sci_o_15,sci_o_16,sci_o_17,sci_o_18,sci_o_19,sci_o_20]


# Reference image is the first image
reference_image = images[0]

o_reference_image = o_images[0]

# Find the location of the maximum value in the reference image
ref_y, ref_x = np.unravel_index(np.argmax(reference_image), reference_image.shape)

oref_y, oref_x = np.unravel_index(np.argmax(o_reference_image), o_reference_image.shape)

    # np.argmax finds the index of the maximum value, and returns it
    # np.unravel_index converts the 1D index from np.argmax() back into 2D (y, x) coordinates based on the original shape of reference_image.


# Initialize a list to store aligned images
aligned_images = []

o_aligned_images = []

# Align each image to the reference image's maximum location
for i, image in enumerate(images):
    # Find the (y, x) coordinates of the maximum value in the current image
    max_y, max_x = np.unravel_index(np.argmax(image), image.shape)

    # Calculate the shift needed to align the max value to the reference position
    y_shift = ref_y - max_y
    x_shift = ref_x - max_x
   
    # Apply the calculated shift to the image
    shifted_image = shift(image, shift=(y_shift, x_shift), mode='nearest')
   
    # Append the shifted image to the aligned_images list
    aligned_images.append(shifted_image)

# Stack aligned images along a new axis and calculate the mean across all aligned images
aligned_mean_image = np.mean(np.stack(aligned_images), axis=0)


plt.figure()
plt.imshow(np.log1p(aligned_mean_image), cmap='gray')
# plt.colorbar()
plt.title("Aligned H-Alpha Images (Mean)")
plt.show()


#%% Aligned_OIII_Images 

for i, image in enumerate(o_images):
    # Find the (y, x) coordinates of the maximum value in the current image
    max_y, max_x = np.unravel_index(np.argmax(image), image.shape)

    # Calculate the shift needed to align the max value to the reference position
    y_shift = ref_y - max_y
    x_shift = ref_x - max_x
   
    # Apply the calculated shift to the image
    shifted_image = shift(image, shift=(y_shift, x_shift), mode='nearest')
   
    # Append the shifted image to the aligned_images list
    o_aligned_images.append(shifted_image)

# Stack aligned images along a new axis and calculate the mean across all aligned images
aligned_o_images = np.mean(np.stack(o_aligned_images), axis=0)*(85/65)


plt.figure()
plt.imshow(np.log1p(aligned_o_images), cmap='gray')
# plt.colorbar()
plt.title("Aligned OIII Images (Mean)")
plt.show()



#%% # Using aperture photometry to remove stars, loading in AperE 


# Use aperE function
def aperE(im, col, row, rad1, rad2, ir1, ir2, or1, or2, Kccd, saturation=np.inf):
    """Original code by Professor Alberto Bolatto, edited by Alyssa Pagan, and
    translated to Python by ChongChong He, further edited by Orion Guiffreda.

    Before using aperE.m, rotate your image using imrotate(im,angle) so the
    major axis of your object is perpendicular or parallel to your x or y axis.
   
    APER(im,col,row,rad1,rad1,ir1,ir2,or1,or2,Kccd) Do aperture photometry of image "im"
    for a star, galaxy or nebula centered at the "row,col" coordinates, For an ellipse
    with a major and minor axis of "rad1,rad2" and an inner sky ellipse with a
    major and minor axis of (ir1,ir2)and outer sky ellipse of "or1,or2" with CCD
    gain of Kccd ADU/electron. Optionally, a 11th parameter can be passed
    with the saturation value for the CCD.
    """

    a, b = im.shape
    xx, yy = np.meshgrid(range(b), range(a))
    ixsrc = ((xx - col) / rad1) ** 2 + ((yy - row) / rad2) ** 2 <= 1
    ixsky = np.logical_and(
        (((xx - col) / or1) ** 2) + (((yy - row) / or2) ** 2) <= 1,
        (((xx - col) / ir1) ** 2) + (((yy - row) / ir2) ** 2) >= 1
    )
    length = max(ixsky.shape)
    sky = np.median(im[ixsky], axis=0)
    imixsrc = im[ixsrc]
    pix = imixsrc - sky
    sig = np.sqrt(imixsrc / Kccd)
    ssig = np.std(im[ixsky]) / np.sqrt(length) / Kccd
    flx = np.sum(pix) / Kccd
    err = np.sqrt(np.sum(sig) ** 2 + ssig ** 2)
    issat = 0
    if max(imixsrc) > saturation:
        issat = 1
    fw = np.copy(or1)
    ix = np.where(
        np.logical_and(
            np.logical_and(
                np.logical_and(xx >= col - 2 * fw, xx <= col + 2 * fw),
                yy >= row - 2 * fw
            ),
            yy <= row + 2 * fw
        )
    )
    aa = np.sum(np.logical_and(xx[0, :] >= col - 2 * fw,
                               xx[0, :] <= col + 2 * fw))
    bb = np.sum(np.logical_and(yy[:, 0] >= row - 2 * fw,
                               yy[:, 0] <= row + 2 * fw))
    px = np.reshape(xx[ix], (bb, aa))
    py = np.reshape(yy[ix], (bb, aa))
    pz = np.reshape(im[ix], (bb, aa))
    plt.figure()
    plt.imshow(pz, extent=[px[0, 0], px[0, -1], py[0, 0], py[-1, 0]])
    plt.tight_layout()
    # if not np.isempty(imixsrc):
    #     np.caxis(np.concatenate((sky, np.array([max(imixsrc)]))))

    p = np.arange(360) * np.pi / 180
    xc = np.cos(p)
    yc = np.sin(p)
    plt.plot(col+rad1*xc, row+rad2*yc, 'w')
    plt.plot(col+ir1*xc, row+ir2*yc, 'r')
    plt.plot(col+or1*xc, row+or2*yc, 'r')
    if issat:
        plt.text(col, row, 'CHECK SATURATION', ha='center', color='w',
                 va='top', fontweight='bold')
        print('At the peak this source has {:0.0f} counts.'.format(
            max(imixsrc)))
        print('Judging by the number of counts, if this is a single exposure the')
        print('source is likely to be saturated. If this is the coadding of many')
        print('short exposures, check in one of them to see if this message appears.')
        print('If it does, you need to flag the source as bad in this output file.')
    plt.tight_layout()
    plt.savefig("aperE_img.pdf")
    return flx, err


# Use DS9 to estimate locations of stars to remove


# On image 1:
    # Star 1: (1052, 729)
    # Star 2: (1056, 650)

#%% AperE

    # Star 1
s1col1 = 1051.6
s1row1 = 727



egain = 1.2999999523162842
kccd = 1/egain

 # Reolace with mean_alligned_image
# Find the flux using aperE
star_1_flux = aperE(
      im = aligned_mean_image,
      row = s1row1,
      col = s1col1,
      rad1 = 8.5,
      rad2 = 6,
      ir1 = 15,
      ir2 = 15,
      or1 = 25,
      or2 = 25,
      Kccd = kccd,
      saturation=np.inf)


s2col1 = 1054
s2row1 = 650

# Star 2:
   
   
# Find the flux using aperE
star_2_flux = aperE(
      im = aligned_mean_image,
      row = s2row1,
      col = s2col1,
      rad1 = 8.5,
      rad2 = 6,
      ir1 = 15,
      ir2 = 15,
      or1 = 25,
      or2 = 25,
      Kccd = kccd,
      saturation=np.inf)


#%% Take Mode of region between inner and outer radii HA stars

from scipy.stats import mode


def annular_mode(im, row, col, inner_radius, outer_radius):
    y, x = np.ogrid[:im.shape[0], :im.shape[1]]
    distance = np.sqrt((x - col)**2 + (y - row)**2)

    # Mask the annular region
    annular_region = (distance >= inner_radius) & (distance <= outer_radius)

    # Extract pixel values in the annular region
    annular_values = im[annular_region]
   
    return mode(annular_values)
   

# Example usage for Star 1's annular region
star_1_annular_mode = annular_mode(aligned_mean_image, s1row1, s1col1, inner_radius=15, outer_radius=25)
print("Star 1 Annular Mode:", star_1_annular_mode[0])



# Example usage for Star 2's annular region
star_2_annular_mode = annular_mode(aligned_mean_image, s2row1, s2col1, inner_radius=15, outer_radius=25)
print("Star 2 Annular Mode:", star_2_annular_mode[0])

#%%

# For HA 
# Replace all ADUS values of Star 1 with annular mode 


# Using poisson statistics to find a minimum brightness value which we count as part of the nebula



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def replace_circle_with_mode(image, center, radius):
    """
    Replace a circular region in a 2D array image with the mode of its values.

    Parameters:
        image (numpy.ndarray): The 2D array (image) to modify.
        center (tuple): Coordinates of the circle's center (y, x).
        radius (int): Radius of the circle.

    Returns:
        numpy.ndarray: The modified 2D array (image).
    """
    # Generate a grid of indices
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    # Calculate the mask for the circular region
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    # Extract the values within the circle
    circular_region = image[mask]
    
    # Compute the mode of the circular region
    circle_mode = mode(circular_region, axis=None).mode[0]
    
    # Replace the circular region with the mode
    image[mask] = circle_mode
    
    return image

# Make a copy of aligned HA image to avoid changing the original image everytime
copied_Ha = aligned_mean_image.copy()


# Parameters for the circular region
circle_center = (s1row1, s1col1)  # Center of the circle (y, x)
circle_radius = 15          # Radius of the circle

circle_2center = (s2row1, s2col1)  # Center of the circle (y, x)
circle_2radius = 15          # Radius of the circle

# Replace the circular region with its mode
modified_image = replace_circle_with_mode(copied_Ha, circle_center, circle_radius)
modified_image2 = replace_circle_with_mode(modified_image, circle_2center, circle_2radius)

# Visualize the original and modified images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(np.log1p(aligned_mean_image), cmap="gray")
ax[0].set_title("Original Ha-Coadded Image")
ax[0].axis("off")



ax[1].imshow(np.log1p(modified_image2), cmap="gray")
ax[1].set_title("Modified Ha - Stars Removed")
ax[1].axis("off")

plt.show()



#%%
# For OIII 
# Replace all ADUS values of Star 1 with annular mode 


# Using poisson statistics to find a minimum brightness value which we count as part of the nebula



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def replace_circle_with_mode(image, center, radius):
    """
    Replace a circular region in a 2D array image with the mode of its values.

    Parameters:
        image (numpy.ndarray): The 2D array (image) to modify.
        center (tuple): Coordinates of the circle's center (y, x).
        radius (int): Radius of the circle.

    Returns:
        numpy.ndarray: The modified 2D array (image).
    """
    # Generate a grid of indices
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    # Calculate the mask for the circular region
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    # Extract the values within the circle
    circular_region = image[mask]
    
    # Compute the mode of the circular region
    circle_mode = mode(circular_region, axis=None).mode[0]
    
    # Replace the circular region with the mode
    image[mask] = circle_mode
    
    return image

# Make a copy of aligned HA image to avoid changing the original image everytime
copied_o = aligned_o_images.copy()


# Parameters for the circular region
circle_center = (s1row1, s1col1)  # Center of the circle (y, x)
circle_radius = 15          # Radius of the circle

circle_2center = (s2row1, s2col1)  # Center of the circle (y, x)
circle_2radius = 15          # Radius of the circle

# Replace the circular region with its mode
modified_o_image = replace_circle_with_mode(copied_o, circle_center, circle_radius)
modified_o_image2 = replace_circle_with_mode(modified_o_image, circle_2center, circle_2radius)

# Visualize the original and modified images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(np.log1p(aligned_o_images), cmap="gray")
ax[0].set_title("Original OIII-Coadded Image")
ax[0].axis("off")



ax[1].imshow(np.log1p(modified_o_image2), cmap="gray")
ax[1].set_title("Modified OIII - Stars Removed")
ax[1].axis("off")

plt.show()












