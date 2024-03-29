from global_config import *
from dataset_specific.config import *

dataset_path = "<path to your chb-mit directory> e.g ./ieeg-swez.ethz.ch/long-term_dataset/"
clean_data_path = "<path to store clean & aligned dataset> e.g ./swecethz_clean"
timeline_info_path = "<path to store record & seizure info for visualization scripts in parent directory e.g ./swecethz-plotinfo>"
label_output_path = "<path to store generated TOML files> e.g ./swecethz_reflabels"
ignore_lst = []
fake_start_datetime = "Fake Record Start Position for special usage such as visualizing timeline in global_config.datetime_format. e.g 2000-01-01 00:00:00"