import os
import sys
import random

from file import read_labels, write_labels


def main(argv=None):
    """Partition the data. 
    """

    #labels_path, out_path = sys.argv[1], sys.argv[2]
    labels_path = 'c:\\CNN-Lungs\\label_final'
    out_path ='c:\\CNN-Lungs\\label_final'

    print('Reading labels files from:', labels_path)
    print('Saving partitioned files in:', out_path)

    in_sample_csv_path = os.path.join(labels_path, 'labels.csv')
    out_of_sample_csv_path = os.path.join(labels_path, 'labels.csv')
    print(in_sample_csv_path)
    print(out_of_sample_csv_path)
    train_ratio = 0.8  # the rest is for validation

    # Read label data.
    in_sample_labels = read_labels(in_sample_csv_path, header=True)
   # out_of_sample_labels = read_labels(out_of_sample_csv_path, header=True)

    # Training and validation sets
    patients = list(in_sample_labels.keys())
    n_train = int(len(patients) * train_ratio)
    

    random.seed(7102)
    random.shuffle(patients)
    train_set = patients[:n_train]
    test_set = patients[n_train:]
    
    # we'll be using part of the training data for validation
    i = int(len(train_set) * 0.1)
    val_set = train_set[:i]
    train_set_final = train_set[i:]


    train_labels = {pid: label for pid, label in in_sample_labels.items() if pid in train_set_final}
    test_labels = {pid: label for pid, label in in_sample_labels.items() if pid in test_set}
    val_labels = {pid: label for pid, label in in_sample_labels.items() if pid in val_set}

    # The test set is the same as the out-of-sample set

    # Save the sets to csv files
    write_labels(train_labels, os.path.join(out_path, 'train.csv'), header=True)
    write_labels(val_labels, os.path.join(out_path, 'validation.csv'), header=True)
    write_labels(test_labels, os.path.join(out_path, 'test.csv'), header=True)

if __name__ == '__main__':
    main()
       
