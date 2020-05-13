import os
import warnings

import numpy as np
import pydicom
import SimpleITK as sitk

from skimage.draw import rectangle_perimeter


def load_all_dicom_images(fnames):
    """
    Load all the DICOM images assocated with this scan and return as list.
    """
    if isinstance(fnames[0], str):
        images = [pydicom.dcmread(fname) for fname in fnames]
    else:
        images = fnames

    zs = [float(img.ImagePositionPatient[-1]) for img in images]
    inums = [float(img.InstanceNumber) for img in images]
    inds = list(range(len(zs)))
    while np.unique(zs).shape[0] != len(inds):
        for i in inds:
            for j in inds:
                if i != j and zs[i] == zs[j]:
                    k = i if inums[i] > inums[j] else j
                    inds.pop(inds.index(k))

    # Prune the duplicates found in the loops above.
    zs = [zs[i] for i in range(len(zs)) if i in inds]
    inums = [inums[i] for i in range(len(inums)) if i in inds]
    images = [images[i] for i in range(len(images)) if i in inds]

    # Sort everything by (now unique) ImagePositionPatient z coordinate.
    sort_inds = np.argsort(inums)
    images = [images[s] for s in sort_inds[::-1]]

    slice_zvals = [zs[s] for s in sort_inds[::-1]]
    z_spacing = np.median(np.diff(slice_zvals))
    slice_spacing = np.append(z_spacing, np.array([images[0].PixelSpacing]))
    # End multiple z clean.

    return images, slice_spacing


def to_volume(images):
    """
    Return the scan as a 3D numpy array volume.
    """
    images, spacing = load_all_dicom_images(images)

    volume = np.zeros((len(images), 512, 512))
    for i in range(len(images)):
        m = float(images[i].RescaleSlope)
        b = float(images[i].RescaleIntercept)

        volume[i] = m * images[i].pixel_array + b
    return volume, spacing


def load_dcm(paths_list):
    filtered_dcm_list = list()
    warnings.filterwarnings('error')
    for path in paths_list:
        dcm = pydicom.dcmread(path, force=True)

        # if hasattr(dcm, 'SeriesDescription'):
        #     print(dcm.SeriesDescription)
        # else:
        #     print('No Series')

        # if hasattr(dcm, 'SliceThinkness'):
        #     print(dcm.SliceThickness)
        # else:
        #     print('No Slice')

        # if (
        #     (not hasattr(dcm, 'SeriesDescription') or dcm.SeriesDescription == 'LDCT 1.5x1.5 IE0.6') and
        #     (not hasattr(dcm, 'SliceThickness') or float(dcm.SliceThickness) <= 2.5)
        # ):
        # if dcm.SeriesDescription == 'LDCT 1.5x1.5 IE0.6' and float(dcm.SliceThickness) <= 2.5:

        try:
            dcm[0x0028, 0x0101].value = 16
        except:
            print('Missing key')

        filtered_dcm_list.append(dcm)

    if filtered_dcm_list:
        return to_volume(filtered_dcm_list)
    else:
        return (None, None)


def load_dcm_broken(path):
    # Load the scans in given folder path
    slices = [
        pydicom.read_file(path + '/' + s)
        for s in os.listdir(path)
        # if s.endswith('dcm')
    ]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_dcm_spacing_broken(scan):
    return np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)


def load_itk_image(filename):

    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def draw_bbox(volume, bboxes, color='r'):
    """Draw bbox into volume"""
    # Check color, Notice that color code must not use 255, which will turn into black
    assert color in ['r', 'g', 'b'], f'{color} not supported'
    color_code = [[255 * (color == 'r')], [255 * (color == 'g')], [255 * (color == 'b')]]

    for (z, y, x, d) in bboxes:
        # Skip if bbox is out of volune
        r = round(d // 2)
        len_z = volume.shape[1]
        start = max(0, z-r)
        end = min(len_z, z+r+1)

        rr, cc = rectangle_perimeter((y-r, x-r), end=(y+r, x+r), shape=volume.shape[2:])
        for slice_idx in range(start, end, 1):
            volume[:, slice_idx, rr, cc] = color_code
