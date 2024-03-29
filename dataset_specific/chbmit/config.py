from global_config import *
from dataset_specific.config import *

dataset_path = "<path to your chb-mit directory> e.g ./physionet.org/files/chbmit/1.0.0/"
clean_data_path = "<path to store clean & aligned dataset> e.g ./chbmit_clean"
timeline_info_path = "<path to store record & seizure info for visualization scripts in parent directory e.g ./chbmit-plotinfo>"
label_output_path = "<path to store generated TOML files> e.g ./chbmit_reflabels"
ignore_lst = ["chb16_18.edf", "chb16_19.edf", "chb17c_13.edf", "chb18_01.edf", "chb19_01.edf", "chb11_01.edf", "chb12_27.edf", "chb12_28.edf", "chb12_29.edf", "chb09_01.edf", "chb15_01.edf"]



