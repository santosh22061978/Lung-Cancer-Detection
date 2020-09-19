# Lung-Cancer-Detection
Develop AI based system to predict the probability of a person being diagnosed with Lung Cancer by looking at their CT Scan Images. 
Use historical CT scan data for ~1000 patients whose CT scans are analyzed  manually by expert radiologist and marked as being cancerous or noncancerous
• Extract voxel data from DICOM into numpy arrays and then perform some low level operations to normalize and resample the data using information in
the DICOM headers.
• Visualize the data in 1D( histogram),2D and 3 D.
• Create segmentation masks that remove all voxel except for the lungs.

About Dataset
• A single slice of CT scan is an image array, typically is taken along the transverse plane of the patient, whose pixels values are measured in
Hounsfield Value. a unit that quantifies the radio density of diverse materials.
• Data for this POC is taken from https://nbia.cancerimagingarchive.net/nbia-search/ and further download the data from LIDC-IDRI and SPIE-AAPM Lung CT Challenge.
• The whole dataset is 150GB, but each examination One CT Scan is only 100MB or so.
• Total 1100 CT scan included in this POC and one CT scan belongs to one patient.
• The first thing we should do is to check to see whether the Houndsfeld Units are properly scaled and represented.
• HU's are useful because it is standardized across all CT scans regardless of the absolute number of photons the scanner detector captured.

Data Visualization

Data extracted from patient for e.g. CT-Training-lc008 . There is lots of air There is some lung. There's an abundance of soft
tissue, mostly muscle, liver, etc, but there's also some fat. There is only a small bit of bone (seen as a tiny sliver of height between 700-1500)
• This observation means that we will need to do significant pre-processing if we want to process lesions in the lung tissue because only a tiny bit of the voxels represent lung.
• Air really only goes to -1000
• "Air," in comparison, appears grey because it has a much higher value. As a result, the lungs and soft tissue have somewhat reduced contrast resolution as well.
• To display the CT in 3D isometric form each slice is resampled in 1x1x1 mm pixels and slices.
• CT slice is typically reconstructed at 512 x 512 voxels, each slice represents approximately 370 mm of data in length and width 


CT Scan Slice Data --> Padding -->Transform Raw Values to HU --> Stack Slices as 3D Array -->Isotropic Resampling -->Extract Lung Area -->3D Array of Lungs -->CNN Classifier

CNN Model Devlopment

First run preprocess_data.py to preprocess the dicom data. It creates 3D multidimensional array for each patient.
This script first creates 3d array in Hounsfield unit, resampled, masking , extract lungs area and then save data in
physical file. This file also prepare one csv(labels.csv) file for each patient class label ( cancerous/non cancerous ).
This file needs to change four variable path (parent_path – patient folder, out_path – to save preprocessed data,
candidates.csv path, labels.csv path ) once run this file successfully output folder path should have one npz file for
each patient and labels.csv file should contain information for each patient class information.
2) Next execute partition_data.py it creates three csv file train set, validation set and test set. To execute the file
successfully provide labels_path - Path containing the labels files and out_path - Path to save the partitioned labels
files. Once input data and their class information available upload all the data on cloud to train the model.
3) To visualize the dicom data run lung_cancer.ipynb python notebook . This notebook will help us to understand data.
4) now input data is ready time to run CNN classifier

CNN Classifier

To classify the patients upon the features transferred from the GMCAE. The output is a single sigmoid unit, and the
network is trained to minimize the Log Loss. Since the model should be able to handle arrays of variable size, a Spatial
Pyramid Pooling layer is used to interface between the convolutional and fully-connected layers.
Setup
Input:
Full 3D array of lung area.
Data augmentation is performed by random rotations/mirroring of the sub-arrays. Since a cube has 48 symmetries
this allows a 48-fold augmentation.
Output:
Probability of being diagnosed with cancer.
First run train_gmcae.py (Gaussian Mixture Convolutional AutoEncoder ). Provide checkpoints_path and dataset_path in
code and refer the document ‘Gaussian Mixture Convolutional AutoEncoder.docx’ for more information. For POC i run file
for 12 epochs only. Basically it should run till minimise the loss approax 100 epochs.

Now high level features are extracted from the encoder and fed to CNN classifier to perform the classification task.
Last train_classifier.py to get probability for each patient. For analysis this script also save prediction information after
training on lung_cancer_pred.csv.
Finally run the LungCancer-Analysis.ipynb python notebook to see the analysis result.
For just load the trained model and find the prediction for run the LungPrediction.ipynb python notebook.

Requirements

Python 3
Keras
tensorflow-gpu
numpy
scipy
pydicom
matplotlib
