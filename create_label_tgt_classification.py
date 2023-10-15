# --------------------------- Configuration Options ----------------------------
# TODO Use argparse to parse these options after refactoring the main script
data_dir = "data_clean"
patient_lst = None  # Use subset of all 24 patients
fs = 256 # Sampling Rate: 256Hz
# ------------------------------------------------------------------------------
# Prediction-specific Parameters
# ------------------------------------------------------------------------------
seizure_occurance_period = 30  # Seizure occurrence period (minutes)
seizure_prediction_horizon = 5  # Seizure prediction horizon (minutes)
# seizure_affected_area = 120 # (minutes) This parameter was split into the following 2 params
seizure_affected_before = 60 # (minutes)
seizure_affected_after = 60 # (minutes)
# ------------------------------------------------------------------------------
extract_ictal_samples = True
extract_preictal_samples = True
extract_interictal_samples = True
# OUTPUT JSON PATH -------------------------------------------------------------
output_dir = "segment_clean"

# -------------------------------- Main Script ---------------------------------
import numpy as np
import os
import datetime
import json

from warnings import warn
if seizure_affected_before < (seizure_occurance_period + seizure_prediction_horizon):
    warn(f'Parameter seizure_affected_before shall be greater-equal than seizure_occurance_period + seizure_prediction_horizon!\n \
           Replace {seizure_affected_before} with {seizure_occurance_period + seizure_prediction_horizon}')
    seizure_affected_before = (seizure_occurance_period + seizure_prediction_horizon)

subdir = os.path.join(output_dir, '%d-%d-%d-%d' % (seizure_occurance_period, # Output dir name by SOP-SPH-SAB-SAA
                                                  seizure_prediction_horizon, 
                                                  seizure_affected_before, 
                                                  seizure_affected_after ))

dt_fmt = '%Y-%m-%d %H:%M:%S'

# Store preictal and non-interictal in a dict
pre_noninter_dict = {}
for pat_id in (patient_lst if patient_lst else range(1, 24+1)):        
    record_info = []; onset_info = []; pre_info = []; noninter_info = []

    pat_pre_segs = []; pat_onset_segs = []; pat_inter_segs = []
    with open(os.path.join(data_dir, 'chb%02d' % pat_id, 'datetime_info.json'), 'r') as f:
        record_lst = json.load(f)
        for record in record_lst:
            edf_name = record['File Name']
            start_str, end_str = record['Record Datetimes']
            start_dt, end_dt = datetime.datetime.strptime(start_str, dt_fmt), datetime.datetime.strptime(end_str, dt_fmt)
            record_info.append((start_dt, end_dt, edf_name))
            for nsz, sz_span in enumerate(record['Seizures']):
                sz_start_sec, sz_end_sec = sz_span
                sz_start_dt, sz_end_dt = start_dt + datetime.timedelta(seconds=sz_start_sec), start_dt + datetime.timedelta(seconds=sz_end_sec)
                if extract_preictal_samples:
                    preStartTime = sz_start_dt - datetime.timedelta(minutes=seizure_occurance_period+seizure_prediction_horizon)
                    preEndTime = sz_start_dt - datetime.timedelta(minutes=seizure_prediction_horizon)
                    postictal_end_dt = onset_info[-1][1] + datetime.timedelta(minutes=seizure_affected_after) if len(onset_info) else datetime.datetime.min # 不能将前一个发作的发作后期也算进当前发作的前期
                    preStartTime = max(preStartTime, postictal_end_dt)
                    if preStartTime < preEndTime:                    
                        # The preictal section falls completely in the current file
                        if preStartTime >= start_dt:
                            pat_pre_segs.append({'Label': 'Pre%d' % (len(onset_info)+1), 
                                                'File': edf_name[:-3] + 'npy', 
                                                'Span': [(preStartTime - start_dt).total_seconds(), (preEndTime - start_dt).total_seconds()]})
                            
                        else:
                            subsegcnt = 0
                            # Backward iteration search for preictal segments in previous records
                            for k in range(len(record_info)-1, -1, -1):
                                if (record_info[k][1] < preStartTime):
                                    break
                                if (record_info[k][0] > preEndTime):
                                    continue
                                sectStart = preStartTime if preStartTime >= record_info[k][0] else record_info[k][0]
                                sectEnd = preEndTime if preEndTime <= record_info[k][1] else record_info[k][1]
                                subsegcnt += 1
                                pat_pre_segs.append({'Label': 'Pre%d-' % (len(onset_info)+1), 
                                                    'File': record_info[k][2][:-3] + 'npy', 
                                                    'Span': [(sectStart - record_info[k][0]).total_seconds(), (sectEnd - record_info[k][0]).total_seconds()]})                                
                            for m in range(subsegcnt):
                                pat_pre_segs[m-subsegcnt]['Label'] = pat_pre_segs[m-subsegcnt]['Label'] + str(m+1) 
                            
                                

                onset_info.append((edf_name, sz_start_dt, sz_end_dt))    
                noninter_info.append((sz_start_dt - datetime.timedelta(minutes=seizure_affected_before), 
                                     sz_end_dt + datetime.timedelta(minutes=seizure_affected_after)))  
                if extract_ictal_samples:
                    pat_onset_segs.append({'Label': 'Onset%d' % len(onset_info), 
                                            'File': edf_name[:-3] + 'npy', 
                                            'Span': [sz_start_sec, sz_end_sec]})                      
    
    if extract_interictal_samples:
        inter_affiliated_onset_cnt = [0] * (len(noninter_info)+1)
        inter_noaffiliated_onset_cnt = 0
        fake_sz_start_dt = record_info[-1][1] # Pretending to have a neighboring onset not recorded after the last file's last recording moment
        fake_sz_inter_end_dt = fake_sz_start_dt - datetime.timedelta(minutes=seizure_affected_before)        
        for dt0, dt1, fn in record_info: # See noninter_info inner structures above
            queue = [(dt0, dt1)] # the list of periods that can be considered as interictal in fn
            already_checked_period_lst = [] # Store the results of BFS
            for nsz, non_dts in enumerate(noninter_info):
                nt0, nt1 = non_dts
                if (nsegs := len(queue)) == 0:
                    break
                for _ in range(nsegs):
                    t0, t1 = queue.pop()
                    if nt0 >= t1:
                        already_checked_period_lst.append((nsz+1, t0, t1))
                    elif nt1 <= t0:
                        queue.append((t0, t1))
                    else:
                        if nt0 > t0:
                            already_checked_period_lst.append((nsz+1, t0, nt0))
                        if nt1 < t1:
                            queue.append((nt1, t1))
            
            for k, t0, t1 in already_checked_period_lst:
                inter_affiliated_onset_cnt[k] += 1
                start_s = (t0 - dt0).total_seconds()
                end_s = (t1 - dt0).total_seconds()

                pat_inter_segs.append({'Label': 'Inter%d-%d' % (k, inter_affiliated_onset_cnt[k]), 
                                    'File': fn[:-3] + 'npy', 
                                    'Span': [start_s, end_s]})
            
            for t0, t1 in queue:
                if t0 >= fake_sz_inter_end_dt:
                    break
                t1 = min(t1, fake_sz_inter_end_dt)
                inter_noaffiliated_onset_cnt += 1
                start_s = (t0 - dt0).total_seconds()
                end_s = (t1 - dt0).total_seconds()
    
                pat_inter_segs.append({'Label': 'Inter+-%d' % (inter_noaffiliated_onset_cnt), 
                                    'File': fn[:-3] + 'npy', 
                                    'Span': [start_s, end_s]})     

    # Merge Results and output json files
    pat_dir = os.path.join(subdir, 'chb%02d' % pat_id)
    os.makedirs(pat_dir, mode=0o755, exist_ok=True)

    pat_all_segs = [*pat_onset_segs, *pat_pre_segs, *pat_inter_segs]
    pat_edf2idx = {t[2]: n for n, t in enumerate(record_info)} # Auxiliary sort usage
    pat_all_segs.sort(key=lambda x: (pat_edf2idx[x['File']], x['Span']))
    pat_all_segs = [{k: (v if k != 'Span' else [round(i*fs) for i in v]) for k, v in d.items()} for d in pat_all_segs] # get idx by multiply Span in seconds with fs written by Bing

    with open(os.path.join(pat_dir, 'segment_info.json'), 'w') as fout:
        json.dump(pat_all_segs, fout, indent=2)

    # print(pat_all_segs, record_info, noninter_info) # For Debug