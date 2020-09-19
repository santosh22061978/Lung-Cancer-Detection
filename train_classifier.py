
import os
import sys
import time
import csv
from collections import OrderedDict

from keras.optimizers import Adam

sys.path.append('../')
from datagen import GeneratorFactory
from model import LungNet


def main():

    # ---------- Data parameters
    checkpoints_path =  '/floyd/home/output'
    gmcae_weights_path = '/floyd/home/scripts/output/weights.10--23588.011333.hdf5'
    data_path = '/floyd/input/traindata/'
    #dataset = 'npz_spacing1x1x1_kernel5_drop0.5p'

    # ---------- Model parameters
    # Encoder parameters. These must be match the ones used in GaussianMixtureCAE.
    nb_filters_per_layer = (64, 128, 256)
    kernel_size = (3, 3, 3)
    padding = 'same'
    batch_normalization = False
    freeze = ['encoder_conv_1-2_0', 'encoder_conv_2-1_0',
              'encoder_conv_1-2_1', 'encoder_conv_2-1_1']

    # Classifier parameters
    spp_nb_bins_per_level = (1, 2, 4)
    n_dense = (1024, 1024)
    dropout_rate = None

    learning_rate = 1e-7
    optimizer = Adam(lr=learning_rate)
    es_patience = 10

    # Training parameters
    volume_resize_factor = 0.3  # TODO: introduced to reduce memory usage. Get either a bigger GPU or rethink GMCAE size
    # batch_size is 1 (full stochastic)
    steps_per_epoch = 689  # Size of train set
    epochs = 15
    validation_steps = 76  # Size of validation set

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log
    print(model_path)

    lungnet = LungNet(nb_filters_per_layer=nb_filters_per_layer, kernel_size=kernel_size, padding=padding,
                      batch_normalization=batch_normalization,
                      spp_nb_bins_per_level=spp_nb_bins_per_level, n_dense=n_dense, dropout_rate=dropout_rate,
                      optimizer=optimizer, es_patience=es_patience,
                      model_path=model_path)

    lungnet.build_model(freeze=freeze)

    lungnet.load_weights_from_file(gmcae_weights_path)

    lungnet.summary()

    # Create data generators
    train_gen_factory = GeneratorFactory(volume_resize_factor=volume_resize_factor, random_rotation=True, random_offset_range=None)
    val_gen_factory = GeneratorFactory(volume_resize_factor=volume_resize_factor, random_rotation=False, random_offset_range=None)

    #dataset_path = os.path.join(data_path, dataset)
    train_gen = train_gen_factory.build_classifier_generator(data_path, 'train')
    val_gen = val_gen_factory.build_classifier_generator(data_path, 'validation')
    test_gen = val_gen_factory.build_predictor(data_path, 'test')

    # Train model
    lungnet.fit_generator(train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                          validation_generator=val_gen, validation_steps=validation_steps)
    
    print("===================================")
    csv_out_path = '/floyd/home/output'
    csv_file = open(os.path.join(csv_out_path, 'lung_cancer_pred.csv'), 'w')
    writer = csv.DictWriter(csv_file, fieldnames=['id', 'cancer'], dialect=csv.excel)#csv_file = open(os.path.join(out_path, 'stage1_submissionX`.csv'), 'w')
    writer.writeheader()

    for x, patient_id in test_gen:
        prob = lungnet.predict(x)
        row_dict = OrderedDict({'id': patient_id, 'cancer': prob})
        writer.writerow(row_dict)
        csv_file.flush()
    #train_gen = train_gen_factory.build_classifier_generator(data_path, 'train')    
    csv_file.close()    
    


if __name__ == '__main__':
    sys.exit(main())