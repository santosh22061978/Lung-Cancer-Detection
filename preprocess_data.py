import os
import sys

import numpy as np
import pandas as pd
from preprocessing import (read_dcm_sequence, check_sequence, make_3d_array,
                                  resample, make_lungs_mask, extract_lungs)
from plot import show_slices
from file import makedir


def main(argv=None):
    """Preprocess the dicom data. 
    """


    parent_path = 'c:\\CNN-Lungs\\dataset'
    out_path = 'c:\\CNN-Lungs\\output'

    new_spacing = (1, 1, 1)  # (2, 2, 2)
    kernel_size = 5  # 3
    slice_drop_prob = 0.005
    save_png = False
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"

    # Make output directory
    dataset_name = 'npz_spacing%s_kernel%d_drop%.1fp' % ('x'.join([str(s) for s in new_spacing]),
                                                         kernel_size,
                                                         100*slice_drop_prob)

    parent_name = os.path.basename(parent_path.rstrip('/'))
    out_path = os.path.join(out_path, parent_name, dataset_name)
    makedir(out_path)
    df_node = pd.read_csv("c:\\CNN-Lungs\\CSVFILES\\candidates.csv")
    print('New spacing:', new_spacing)
    print('Kernel size:', kernel_size)
    print('Slice drop probability:', slice_drop_prob)
    print('Output path:', out_path)

    patients = [os.path.basename(path) for path in os.listdir(parent_path)]

    for patient_id in patients:
        
        print(patient_id)
        path1 = parent_path+"\\"+str(patient_id)
        path2_file = [os.path.basename(path) for path in os.listdir(path1)]
        path3 = path1+"\\"+str(path2_file[0])
        path3_file = [os.path.basename(path) for path in os.listdir(path3)]
        path4 = path3+"\\"+str(path3_file[0])
        try:
            # Get slice sequence
            print("-----------++++++----")
            dcm_seq = read_dcm_sequence(patient_id, path4)
            #first update the data set with series id with pateint id so it should have user friendly mapping
            df_node1 = df_node[df_node.seriesuid == dcm_seq[0].SeriesInstanceUID ]
            df_node1.patient_id = patient_id
            df_node.update(df_node1)
           # print(dcm_seq)
            #dcm_seq=['E:\\CNN-Lungs\\dataset\\4-HIGH RES-47.17\\000000.dcm', 'E:\\CNN-Lungs\\dataset\\4-HIGH RES-47.17\\000001.dcm']
            print("-----------------")
            dcm_seq = check_sequence(dcm_seq)

            # Stack into 3D array in Hounsfield Units
            array, spacing = make_3d_array(dcm_seq)

            # Resample
            array_resampled = resample(array, spacing, new_spacing)

            # Extract lungs
            mask_lungs, thres = make_lungs_mask(array_resampled, kernel_size=kernel_size)
            array_lungs, box = extract_lungs(array_resampled, mask_lungs, slice_drop_prob)
            
            # Outputs
            if save_png:
                png_filename = os.path.join(out_path, patient_id + '.png')
                show_slices(array_lungs, filename=png_filename, every=5)

            npz_filename = os.path.join(out_path, patient_id + '.npz')
            np.savez_compressed(npz_filename, array_lungs=array_lungs)
#            print(patient_id)
#            print(dcm_seq[0].SeriesInstanceUID)
        except Exception as ex:
            message = template.format(type(ex).__name__, ex.args)
            print('fail', patient_id, message)

            continue

        else:
            print(
                'success',
                patient_id, thres, box,
                array.shape, array_resampled.shape, array_lungs.shape
                )
    # data preparation for patient and their class label( target whether they have cancer or not        
    df_node.to_csv("c:\\CNN-Lungs\\CSVFILES\\candidates.csv",index=False)
    df_node = pd.read_csv("c:\\CNN-Lungs\\CSVFILES\\candidates.csv")
    #print(df_node.info())
    #print(df_node.groupby(['patient_id'], sort=False)['class'].max())
    #df_patient = df_node.groupby(['patient_id'])['class'].transform(max) == df_node['class']
    df_patient = (df_node
                      .groupby(['patient_id'])[['class']]
                      .max()
                      .reset_index()    
                 )
    #print(df_node[df_patient])
    df_patient.to_csv("c:\\CNN-Lungs\\CSVFILES\\labels.csv",index=False)              
if __name__ == '__main__':
    main()