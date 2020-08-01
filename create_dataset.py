from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from data_processing.urban_sound_8K import UrbanSound8K
from data_processing.dataset import Dataset
import warnings

warnings.filterwarnings(action='ignore')

mozilla_basepath = 'Speech-Dataset-in-Hindi-Language/TrainingAudio'
urbansound_basepath = '/Noise'

mcv = MozillaCommonVoiceDataset(mozilla_basepath, val_dataset_size=10)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()
us8K = UrbanSound8K(urbansound_basepath, val_dataset_size=1)
noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}
# print(len(noise_train_filenames),len(noise_val_filenames))
val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=1)
# input("Done")
train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=9)
input("done")
## Create Test Set
# clean_test_filenames = mcv.get_test_filenames()
# noise_test_filenames = us8K.get_test_filenames()
#
# test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
# test_dataset.create_tf_record(prefix='test', subset_size=100, parallel=False)
