# import module for azimuthal integration
# import module for X-ray images opening
import pyFAI, fabio
# import module for displaying image
import matplotlib
from matplotlib import pyplot as plt
# import string module
import string
# import time module
import time
# import jupyter
from pyFAI.gui import jupyter
# import os
import os
# import numpy
import numpy as np
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import astroscrappy

def als_baseline(y, lam=1e3, p=0.001, niter=1000):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())  # penalty matrix

    w = np.ones(L)
    for i in range(niter):
        W = sparse.diags(w, 0)
        Z = W + D
        z = spsolve(Z.tocsc(), w * y)  # Convert to CSC format
        w = p * (y > z) + (1 - p) * (y < z)
    return z

# define path to h5 master file
path_to_master_file=r"E:\100225_backup\GIWAXS_measurements\Ag_Xray_source\Erik_battery_backup\151025_good_one_rhd_1st_battery\14-10-2025\RHD_battery\x_-4.10mm,z_-3.5mm,w_4mm-24hours\series_398_master.h5"
# open h5 master file with fabio
img = fabio.open(path_to_master_file)
# path to calibration poni file
path_to_calib_poni=r"E:\100225_backup\GIWAXS_measurements\Ag_Xray_source\Erik_battery_backup\031025_good_one\02-10-2025\calibration_LaB6\x_-6.10mm,z_-22mm,w_4mm\LaB6.poni"
# load pyFAI calibration poni file
ai = pyFAI.load(path_to_calib_poni)
# specify size of images
# number of pixels in x /  horizontal dierction or number of columns
dim_x = 1028
# number of pixels in y / vertical direction or number of rows
dim_y = 512
# define pixel size - horiozntal, in meters
pixel_size_x=75e-6
# define pixel size - vertical, in meters
pixel_size_y=75e-6
# flip horizontal center - poni2 in horiozntal direction
ai.poni2=dim_x*pixel_size_x - ai.poni2
ai.rot1=(-1.0)*ai.rot1
##print(ai.rot1)
# path to mask file
path_to_mask_file=r"E:\100225_backup\GIWAXS_measurements\Ag_Xray_source\Erik_battery_backup\031025_good_one\02-10-2025\calibration_LaB6\x_-6.10mm,z_-22mm,w_4mm\LaB6.edf"
# tell to program if you want to use mask, it can be True or False boolean value
mask_switch = True
if (mask_switch == True):
    mask_img = fabio.open(path_to_mask_file)
else:
    print("Mask is not used")
# flip mask data
mask_flipped = np.fliplr(mask_img.data)
# specify root name of single image
image_root_name = "image"
# specify image name separator
image_name_separator = "_"
# specify image name extension
twod_image_name_extension = ".edf"
# number of digits in image numbering
no_image_digits = 5
# specify number of points in radial cut
no_points_in_radial_cut = 1000
# specify units in which you want to perform azimuthal cut ("2th_deg" or "r_mm" or "q_nm^-1")
unit_of_radial_cut = "q_nm^-1"
# aziumthal cut name extension specified as dat file
radial_cut_name_extension = ".txt"
### specify variance ndarray, array containing the variance of the data, use None if you don't know
##variance_selector = None
# specify error model for calculation variance e.g. None or "poisson" or "azimuthal" (use defaultly "poisson")
error_model_selector = "poisson"
# specify radial range, ((float, float), optional), if not used, use None or ( , )
# the lower and upper range of the radial unit. If not provided, range is simply (min, max). Values outside the range are ignored.
radial_range_selector = [2.3 , 74.0]
# specify azimuthal range, ((float, float), optional), if not used, use None or ( , )
# the lower and upper range of the azimuthal angle in degree. If not provided, range is simply (min, max). Values outside the range are ignored.
azimuth_range_selector  = [-60 , 60.0]
# specify polarization factor, if experiment was done in laboratory, use None
# polarization factor between -1 (vertical) and +1 (horizontal). 0 for circular polarization or random, None for no correction, True for using the former correction
polarization_factor_selector = 0.0
### specify float value of a normalization monitor, use None (no correction)
##normalization_factor_selector = None
### dark (ndarray) – dark noise image
##dark_selector = None
### flat (ndarray) – flat field image
flat_selector = None
# method (IntegrationMethod) – IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
method_selector = ("full", "histogram", "cython")
# input data for 2D integration
# number of radial points
npt_rad_2D = no_points_in_radial_cut
# number of azimuthal points or chi points
npt_azim_2D = 180
# extension of 2D integration image
integ_2D_name_extension = ".edf"
# select radial range for 2D integration
radial_range_selector_2D = [0.23 , 7.40]
# select azimuthal range for 2D integration
azimuth_range_selector_2D = [0, 360]
# select method for 2D integration
method_selector_2D = "bbox"
# select unit for radial 2D integration
# Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
unit_selector_2D = "q_A^-1"
# define number image calculations
no_image_calc = img.nframes
# create numpy array to save time-resolved q-cuts
intensity_1d_integs = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_q_coord = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_tth_coord = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_sigma = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_integs_cosmic = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_baseline = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_baseline_cosmic = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_corrected = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
intensity_1d_corrected_cosmic = np.zeros((no_points_in_radial_cut, no_image_calc), dtype=float)
# path to output_folder
path_to_output_folder=r"e:\100225_backup\GIWAXS_measurements\Ag_Xray_source\Erik_battery_backup\151025_good_one_rhd_1st_battery\14-10-2025\RHD_battery\output_folder"
# create output sub folders for radial cuts and 2D integartions
path_to_output_folder_radial_cuts = os.path.join(path_to_output_folder, "radial_cuts")
path_to_output_folder_radial_cuts_cosmic = os.path.join(path_to_output_folder, "radial_cuts_cosmic")
# paths to 1D integrations 
path_to_output_folder_1D_integs_intensity_txt = os.path.join(path_to_output_folder, "1D_integrations_intensity_txt")
path_to_output_folder_1D_integs_q_coord_txt = os.path.join(path_to_output_folder, "1D_integrations_q_coord_txt")
path_to_output_folder_1D_integs_tth_coord_txt = os.path.join(path_to_output_folder, "1D_integrations_twotheta_coord_txt")
path_to_output_folder_1D_integs_sigma_txt = os.path.join(path_to_output_folder, "1D_integrations_sigma_txt")
path_to_output_folder_1D_baseline_txt = os.path.join(path_to_output_folder, "1D_integrations_baseline_txt")
path_to_output_folder_1D_baseline_cosmic_txt = os.path.join(path_to_output_folder, "1D_integrations_baseline_cosmic_txt")
path_to_output_folder_1D_corrected_txt = os.path.join(path_to_output_folder, "1D_integrations_corrected_txt")
path_to_output_folder_1D_corrected_cosmic_txt = os.path.join(path_to_output_folder, "1D_integrations_corrected_cosmic_txt")
# paths to 2D integrations
path_to_output_folder_2D_integs_images = os.path.join(path_to_output_folder, "2D_integrations_images") 
path_to_output_folder_2D_integs_intensity_txt = os.path.join(path_to_output_folder, "2D_integrations_intensity_txt")
path_to_output_folder_2D_integs_radial_coord_txt = os.path.join(path_to_output_folder, "2D_integrations_radial_coord_txt")
path_to_output_folder_2D_integs_azimut_coord_txt = os.path.join(path_to_output_folder, "2D_integrations_azimut_coord_txt")
# create new output folders
os.mkdir(path_to_output_folder_radial_cuts)
os.mkdir(path_to_output_folder_radial_cuts_cosmic)
# create new output folders for 1D integrations
os.mkdir(path_to_output_folder_1D_integs_intensity_txt)
os.mkdir(path_to_output_folder_1D_integs_q_coord_txt)
os.mkdir(path_to_output_folder_1D_integs_tth_coord_txt)
os.mkdir(path_to_output_folder_1D_integs_sigma_txt)
os.mkdir(path_to_output_folder_1D_baseline_txt)
os.mkdir(path_to_output_folder_1D_baseline_cosmic_txt)
os.mkdir(path_to_output_folder_1D_corrected_txt)
os.mkdir(path_to_output_folder_1D_corrected_cosmic_txt)
# create new output folders for 2D integrations
os.mkdir(path_to_output_folder_2D_integs_images)
os.mkdir(path_to_output_folder_2D_integs_intensity_txt)
os.mkdir(path_to_output_folder_2D_integs_radial_coord_txt)
os.mkdir(path_to_output_folder_2D_integs_azimut_coord_txt) 
# open all images in h5 master file
for frame_id in range(0, img.nframes):
    # print frame id
    print("The frame id is: ", frame_id)
    # calculate current image number
    image_number_current = frame_id
    # cast time binning current image number to string
    image_number_current_str = str(image_number_current)
    # right paddding of string with zeros
    image_number_current_str = image_number_current_str.rjust(no_image_digits, '0')
    # generate full current image name
    image_name_full = image_root_name + image_name_separator + image_number_current_str
    # flip image data
    current_image = np.fliplr(img.get_frame(frame_id).data)
    # Detect cosmic rays
    mask, cosmic_image = astroscrappy.detect_cosmics(current_image, sigclip=4.5, gain=1.0, readnoise=6.5, niter=4)
    # perform q-cut or 2theta-cut
    # specify radial cut name
    radial_cut_name = image_root_name + "_radial_cut" + image_name_separator + image_number_current_str
    # specify cosmic radial cut name
    radial_cut_name_cosmic = image_root_name + "_radial_cut_cosmic" + image_name_separator + image_number_current_str
    # specify path to radial cut
    path_to_radial_cut = os.path.join(path_to_output_folder_radial_cuts, radial_cut_name + radial_cut_name_extension)
    # specify path to cosmic radial cut
    path_to_radial_cut_cosmic = os.path.join(path_to_output_folder_radial_cuts_cosmic, radial_cut_name_cosmic + radial_cut_name_extension)
    if (mask_switch == True):
        if (unit_of_radial_cut == "q_nm^-1"):
            q, intensity, sigma = ai.integrate1d_ng(current_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    mask = mask_flipped,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut)
            q, intensity_cosmic, sigma = ai.integrate1d_ng(cosmic_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    mask = mask_flipped,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut_cosmic)
            intensity_1d_integs[:, frame_id] = intensity
            intensity_1d_integs_cosmic[:, frame_id] = intensity_cosmic
            intensity_1d_integs_q_coord[:, frame_id] = q
            intensity_1d_integs_sigma[:, frame_id] = sigma
        else:
            twotheta, intensity, sigma = ai.integrate1d_ng(current_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           mask = mask_flipped,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_radial_cut)
            twotheta, intensity_cosmic, sigma = ai.integrate1d_ng(cosmic_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           mask = mask_flipped,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_radial_cut_cosmic)
            intensity_1d_integs[:, frame_id] = intensity
            intensity_1d_integs_cosmic[:, frame_id] = intensity_cosmic
            intensity_1d_integs_tth_coord[:, frame_id] = twotheta
            intensity_1d_integs_sigma[:, frame_id] = sigma
    else:
        if (unit_of_radial_cut == "q_nm^-1"):
            q, intensity, sigma = ai.integrate1d_ng(current_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut)
            q, intensity_cosmic, sigma = ai.integrate1d_ng(cosmic_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut, error_model=error_model_selector,
                                                    radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                    polarization_factor = polarization_factor_selector, 
                                                    method = method_selector, 
                                                    filename=path_to_radial_cut_cosmic)
            intensity_1d_integs[:, frame_id] = intensity
            intensity_1d_integs_cosmic[:, frame_id] = intensity_cosmic
            intensity_1d_integs_q_coord[:, frame_id] = q
            intensity_1d_integs_sigma[:, frame_id] = sigma
        else:
            twotheta, intensity, sigma = ai.integrate1d_ng(current_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_radial_cut)
            twotheta, intensity_cosmic, sigma = ai.integrate1d_ng(cosmic_image, npt = no_points_in_radial_cut, unit=unit_of_radial_cut,  error_model=error_model_selector,
                                                           radial_range = radial_range_selector, azimuth_range = azimuth_range_selector,
                                                           polarization_factor = polarization_factor_selector, 
                                                           method = method_selector, 
                                                           filename=path_to_radial_cut_cosmic)
            intensity_1d_integs[:, frame_id] = intensity
            intensity_1d_integs_cosmic[:, frame_id] = intensity_cosmic
            intensity_1d_integs_tth_coord[:, frame_id] = twotheta
            intensity_1d_integs_sigma[:, frame_id] = sigma
    # do baseline subtraction
    baseline = als_baseline(intensity_1d_integs[:, frame_id])
    corrected = intensity_1d_integs[:, frame_id] - baseline
    baseline_cosmic = als_baseline(intensity_1d_integs_cosmic[:, frame_id])
    corrected_cosmic = intensity_1d_integs_cosmic[:, frame_id] - baseline_cosmic
    # save baseline and corrected data
    intensity_1d_baseline[:, frame_id] = baseline
    intensity_1d_corrected[:, frame_id] = corrected
    intensity_1d_baseline_cosmic[:, frame_id] = baseline_cosmic
    intensity_1d_corrected_cosmic[:, frame_id] = corrected_cosmic
    # plot background subtraction
    if (unit_of_radial_cut == "q_nm^-1"):
        plt.plot(intensity_1d_integs_q_coord[:, frame_id], intensity_1d_integs[:, frame_id], label='Original')
        plt.plot(intensity_1d_integs_q_coord[:, frame_id], intensity_1d_baseline[:, frame_id], label='Baseline')
        plt.plot(intensity_1d_integs_q_coord[:, frame_id], intensity_1d_corrected[:, frame_id], label='Corrected')
        plt.plot(intensity_1d_integs_q_coord[:, frame_id], intensity_1d_corrected_cosmic[:, frame_id], label='Corrected cosmic')
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.plot(intensity_1d_integs_tth_coord[:, frame_id], intensity_1d_integs[:, frame_id], label='Original')
        plt.plot(intensity_1d_integs_tth_coord[:, frame_id], intensity_1d_baseline[:, frame_id], label='Baseline')
        plt.plot(intensity_1d_integs_tth_coord[:, frame_id], intensity_1d_corrected[:, frame_id], label='Corrected')
        plt.plot(intensity_1d_integs_tth_coord[:, frame_id], intensity_1d_corrected_cosmic[:, frame_id], label='Corrected cosmic')
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    # perorm 2D integration in q vs chi plot
    # specify path to 2D integration image
    path_to_integ_2D_images = os.path.join(path_to_output_folder_2D_integs_images, image_name_full + twod_image_name_extension)
    path_to_2D_integs_intensity_txt = os.path.join(path_to_output_folder_2D_integs_intensity_txt, image_name_full + ".txt")
    path_to_2D_integs_radial_coord_txt = os.path.join(path_to_output_folder_2D_integs_radial_coord_txt, image_name_full + "_radial.txt")
    path_to_2D_integs_azimut_coord_txt = os.path.join(path_to_output_folder_2D_integs_azimut_coord_txt, image_name_full + "_azimut.txt")
    if (mask_switch == True):
        intensity_integ_2D, radial, azimuthal, sigma_integ_2D = ai.integrate2d_ng(cosmic_image, npt_rad = npt_rad_2D, npt_azim = npt_azim_2D, filename = path_to_integ_2D_images,
                       error_model=error_model_selector,
                       mask = mask_flipped,  
                       polarization_factor = polarization_factor_selector,
                       method = method_selector_2D, unit = unit_selector_2D)
        # save all data from 2D integartions as text files
        np.savetxt(path_to_2D_integs_intensity_txt, intensity_integ_2D, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_radial_coord_txt, radial, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_azimut_coord_txt, azimuthal, delimiter='\t', newline='\n')
    else:
        intensity_integ_2D, radial, azimuthal = ai.integrate2d_ng(cosmic_image, npt_rad = npt_rad_2D, npt_azim = npt_azim_2D, filename = path_to_integ_2D_images,
                       error_model=error_model_selector,
                       polarization_factor = polarization_factor_selector,
                       method = method_selector_2D, unit = unit_selector_2D)
        # save all data from 2D integartions as text files
        np.savetxt(path_to_2D_integs_intensity_txt, intensity_integ_2D, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_radial_coord_txt, radial, delimiter='\t', newline='\n')
        np.savetxt(path_to_2D_integs_azimut_coord_txt, azimuthal, delimiter='\t', newline='\n')
# generate paths for 1D or radial integrations
path_to_1D_integs_intensity_txt = os.path.join(path_to_output_folder_1D_integs_intensity_txt, image_root_name + "_intensity.txt")
path_to_1D_integs_q_coord_txt = os.path.join(path_to_output_folder_1D_integs_q_coord_txt, image_root_name + "_q.txt")
path_to_1D_integs_tth_coord_txt = os.path.join(path_to_output_folder_1D_integs_tth_coord_txt, image_root_name + "_2theta.txt")
path_to_1D_integs_sigma_txt = os.path.join(path_to_output_folder_1D_integs_sigma_txt, image_root_name + "_sigma.txt")
path_to_1D_baseline_txt = os.path.join(path_to_output_folder_1D_baseline_txt, image_root_name + "_baseline.txt")
path_to_1D_baseline_cosmic_txt = os.path.join(path_to_output_folder_1D_baseline_cosmic_txt, image_root_name + "_baseline_cosmic.txt")
path_to_1D_corrected_txt = os.path.join(path_to_output_folder_1D_corrected_txt, image_root_name + "_corrected.txt")
path_to_1D_corrected_cosmic_txt = os.path.join(path_to_output_folder_1D_corrected_cosmic_txt, image_root_name + "_corrected_cosmic.txt")
# save radial cuts in matrix form
if (unit_of_radial_cut == "q_nm^-1"):
    # save all data from 1D integartions as text files
    np.savetxt(path_to_1D_integs_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_integs_q_coord_txt, intensity_1d_integs_q_coord, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_integs_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_baseline_txt, intensity_1d_baseline, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_baseline_cosmic_txt, intensity_1d_baseline_cosmic, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_corrected_txt, intensity_1d_corrected, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_corrected_cosmic_txt, intensity_1d_corrected_cosmic, delimiter='\t', newline='\n')
else:
    # save all data from 1D integartions as text files
    np.savetxt(path_to_1D_integs_intensity_txt, intensity_1d_integs, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_integs_tth_coord_txt, intensity_1d_integs_tth_coord, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_integs_sigma_txt, intensity_1d_integs_sigma, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_baseline_txt, intensity_1d_baseline, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_baseline_cosmic_txt, intensity_1d_baseline_cosmic, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_corrected_txt, intensity_1d_corrected, delimiter='\t', newline='\n')
    np.savetxt(path_to_1D_corrected_cosmic_txt, intensity_1d_corrected_cosmic, delimiter='\t', newline='\n')
