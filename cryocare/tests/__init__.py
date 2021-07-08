

from pyworkflow.tests import DataSet
DataSet(name='cryocare', folder='cryocare',
        files={
            'rec_even_odd_tomos_dir': 'Tomos_EvenOdd_Reconstructed',
            'tomo_even': 'Tomos_EvenOdd_Reconstructed/Tomo110_Even_bin6.mrc',
            'tomo_odd': 'Tomos_EvenOdd_Reconstructed/Tomo110_Odd_bin6.mrc',
            'model_dir': 'Training_Model',
            'training_data_dir': 'Training_Data',
            'train_data_file': 'Training_Data/train_data.npz',
            'validation_data_file': 'Training_Data/val_data.npz',
            'training_data_conf_dir': 'Training_Data_Config',
            'training_data_conf': 'Training_Data_Config/training_data_config'
        })
