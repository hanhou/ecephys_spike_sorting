import os
import shutil
import subprocess
import numpy as np

from helpers import SpikeGLX_utils
from helpers import log_from_json
from helpers import run_one_probe
from create_input_json import createInputJson

# script to run CatGT, KS2, postprocessing and TPrime on data collected using
# SpikeGLX. The construction of the paths assumes data was saved with
# "Folder per probe" selected (probes stored in separate folders) AND
# that CatGT is run with the -out_prb_fld option

# -------------------------------
# -------------------------------
# User input -- Edit this section
# -------------------------------
# -------------------------------

# brain region specific params
# can add a new brain region by adding the key and value for each param
# can add new parameters -- any that are taken by create_input_json --
# by adding a new dictionary with entries for each region and setting the
# according to the new dictionary in the loop to that created json files.
ksTh_dict = {'default': '[10,4]', 'cortex': '[10,4]', 'striatum': '[10,4]', 'medulla': '[10,4]', 'midbrain': '[10,4]',
             'thalamus': '[10,4]'}
refPerMS_dict = {'default': 2.0, 'cortex': 2.0, 'striatum': 2.0, 'medulla': 1.0, 'midbrain': 1.5, 'thalamus': 1.5}

# -----------
# Input data
# -----------
# Name for log file for this pipeline run. Log file will be saved in the
# output destination directory catGT_dest
# If this file exists, new run data is appended to it
logName = 'foraging_ephys_log.csv'

# Raw data directory = npx_directory
# run_specs = name, gate, trigger and probes to process
npx_directory = r'F:\ephys_raw\HH09'

# Each run_spec is a list of 4 strings:
#   undecorated run name (no g/t specifier, the run field in CatGT)
#   gate index, as a string (e.g. '0')
#   triggers to process/concatenate, as a string e.g. '0,400', '0,0 for a single file
#           can replace first limit with 'start', last with 'end'; 'start,end'
#           will concatenate all trials in the probe folder
#   probes to process, as a string, e.g. '0', '0,3', '0:3'
#   brain regions, list of strings, one per probe, to set region specific params
#           these strings must match a key in the param dictionaries above.

run_specs = [

            ['HH09_S06_20210609', '0', '0,0', '0:1', ['cortex', 'cortex']],
            ['HH09_S07_20210611_surface', '0', '0,0', '0:1', ['cortex', 'cortex']],
            ['HH09_S07_20210611', '0', '0,0', '0:1', ['cortex', 'cortex']],
            ['HH09_S08_20210612', '0', '0,0', '0:1', ['cortex', 'cortex']],
            ['HH09_S09_20210613', '0', '0,0', '0:1', ['cortex', 'cortex']],
            ['HH09_S10_20210614', '0', '0,0', '0', ['midbrain']],

]

# ------------------
# Output destination
# ------------------
# Set to an existing directory; all output will be written here.
# Output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin
catGT_dest = r'I:\catGT\HH09'

# ------------
# CatGT params
# ------------
run_CatGT = True  # set to False to sort/process previously processed data.

# CAR mode for CatGT. Must be equal to 'None', 'gbldmx', or 'loccar'
car_mode = 'loccar'
# car_mode = 'gbldmx'
# inner and outer radii, in um for local comman average reference, if used
loccar_min = 40
loccar_max = 160

# CatGT commands for bandpass filtering, artifact correction, and zero filling
# Note 1: directory naming in this script requires -prb_fld and -out_prb_fld
# Note 2: this command line includes specification of edge extraction
# see CatGT readme for details
# these parameters will be used for all runs
catGT_cmd_string = '-prb_fld -out_prb_fld -aphipass=300 -aplopass=9000 -tshift -gfix=0.4,0.10,0.02'
# catGT_cmd_string = '-prb_fld -out_prb_fld -aphipass=300'

lfp_needed = True  # Whether to process LFP
lfp_string = ' -lfhipass=0.1 -lflopass=1000'
if lfp_needed:
    catGT_cmd_string += lfp_string

ni_present = True
n_XAs = 3  # Number of analog channels
ni_sync = f'XD={n_XAs},0,500'      # Sync channel in NIDQ

ni_extract_string = (
                    f'-{ni_sync} '    # Sync pulse in nidq file: word {a=1 in Dave's rig now}, threshold {b=1}V, min {c=3}V, pulse width {d=500} ms
                    f'-XD={n_XAs},1,0 '   # bpod-trial start
                    f'-XD={n_XAs},2,1 '   # actual bit code: 1 ms width
                    f'-XD={n_XAs},2,1.5 '  # start of bit code: word 4, bit 1, 1.5 ms
                    f'-XD={n_XAs},2,2 '    # Choice_L: 2    ms width
                    f'-XD={n_XAs},2,2.5 '  # Choice_R: 2.5 ms
                    f'-XD={n_XAs},2,10 '   # go cue: 10 ms width
                    f'-XD={n_XAs},2,20 '   # reward: 20 ms width
                    f'-XD={n_XAs},2,30 '   # ITI start: 30 ms
                    f'-XD={n_XAs},4,0 '   # Zaber movement (rising)
                    f'-iXD={n_XAs},4,0 '  # Zaber movement (falling)
                    f'-XD={n_XAs},5,0 '   # Camera 1 (300 Hz)
                    f'-XD={n_XAs},6,0 '   # Camera 2 (300 Hz)
                    f'-XD={n_XAs},7,0 '   # Camera 3 (300 Hz)
                    )

# ----------------------
# KS2 or KS25 parameters
# ----------------------
# parameters that will be constant for all recordings
# Template ekmplate radius and whitening, which are specified in um, will be
# translated into sites using the probe geometry.
ks_remDup = 0
ks_saveRez = 1
ks_copy_fproc = 0
ks_templateRadius_um = 163
ks_whiteningRadius_um = 163
ks_minfr_goodchannels = 0.1

# ----------------------
# C_Waves snr radius, um
# ----------------------
c_Waves_snr_um = 160

# ----------------------
# psth_events parameters
# ----------------------
# extract param string for psth events -- copy the CatGT params used to extract
# events that should be exported with the phy output for PSTH plots
# If not using, remove psth_events from the list of modules
event_ex_param_str = f'XD={n_XAs},2,10'  # Go cue

# -----------------
# TPrime parameters
# -----------------
runTPrime = True  # set to False if not using TPrime
sync_period = 1.0  # true for SYNC wave generated by imec basestation
toStream_sync_params =  'SY=0,-1,6,500'  # copy from the CatGT command line, no spaces
niStream_sync_params =  ni_sync   # copy from the CatGT comman line, set to None if no Aux data, no spaces


# ---------------
# Modules List
# ---------------
# List of modules to run per probe; CatGT and TPrime are called once for each run.
modules = [
    'kilosort_helper',
    'kilosort_postprocessing',
    'noise_templates',
    'psth_events',
    'mean_waveforms',
    'quality_metrics'
]

json_directory = r'I:\json_file'

# -----------------------
# -----------------------
# End of user input
# -----------------------
# -----------------------

# delete the existing CatGT.log
try:
    os.remove('CatGT.log')
except OSError:
    pass

# delete existing Tprime.log
try:
    os.remove('Tprime.log')
except OSError:
    pass

# delete existing C_waves.log
try:
    os.remove('C_Waves.log')
except OSError:
    pass

# check for existence of log file, create if not there
logFullPath = os.path.join(catGT_dest, logName)
if not os.path.isfile(logFullPath):
    # create the log file, write header
    log_from_json.writeHeader(logFullPath)

for spec in run_specs:

    session_id = spec[0]

    # Make list of probes from the probe string
    prb_list = SpikeGLX_utils.ParseProbeStr(spec[3])

    # build path to the first probe folder; look into that folder
    # to determine the range of trials if the user specified t limits as
    # start and end
    run_folder_name = spec[0] + '_g' + spec[1]
    prb0_fld_name = run_folder_name + '_imec' + prb_list[0]
    prb0_fld = os.path.join(npx_directory, run_folder_name, prb0_fld_name)
    first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(spec[2], prb_list[0], spec[1], prb0_fld)
    trigger_str = repr(first_trig) + ',' + repr(last_trig)

    # loop over all probes to build json files of input parameters
    # initalize lists for input and output json files
    catGT_input_json = []
    catGT_output_json = []
    module_input_json = []
    module_output_json = []
    session_id = []
    data_directory = []

    # first loop over probes creates json files containing parameters for
    # both preprocessing (CatGt) and sorting + postprocessing

    for i, prb in enumerate(prb_list):

        # create CatGT command for this probe
        print('Creating json file for CatGT on probe: ' + prb)
        # Run CatGT
        catGT_input_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-input.json'))
        catGT_output_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-output.json'))

        # build extract string for SYNC channel for this probe
        sync_extract = '-SY=' + prb + ',-1,6,500'

        # if this is the first probe proceessed, process the ni stream with it
        if i == 0 and ni_present:
            catGT_stream_string = '-ap -ni' + (' -lf' if lfp_needed else '')
            extract_string = sync_extract + ' ' + ni_extract_string
        else:
            catGT_stream_string = '-ap' + (' -lf' if lfp_needed else '')
            extract_string = sync_extract

        # build name of first trial to be concatenated/processed;
        # allows reaidng of the metadata
        run_str = spec[0] + '_g' + spec[1]
        run_folder = run_str
        prb_folder = run_str + '_imec' + prb
        input_data_directory = os.path.join(npx_directory, run_folder, prb_folder)
        fileName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(input_data_directory, fileName)
        metaName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.meta'
        input_meta_fullpath = os.path.join(input_data_directory, metaName)

        print(input_meta_fullpath)

        info = createInputJson(catGT_input_json[i], npx_directory=npx_directory,
                               continuous_file=continuous_file,
                               kilosort_output_directory=catGT_dest,
                               spikeGLX_data=True,
                               input_meta_path=input_meta_fullpath,
                               catGT_run_name=spec[0],
                               gate_string=spec[1],
                               trigger_string=trigger_str,
                               probe_string=prb,
                               catGT_stream_string=catGT_stream_string,
                               catGT_car_mode=car_mode,
                               catGT_loccar_min_um=loccar_min,
                               catGT_loccar_max_um=loccar_max,
                               catGT_cmd_string=catGT_cmd_string + ' ' + extract_string,
                               extracted_data_directory=catGT_dest
                               )

        # create json files for the other modules
        session_id.append(spec[0] + '_imec' + prb)

        module_input_json.append(os.path.join(json_directory, session_id[i] + '-input.json'))

        # location of the binary created by CatGT, using -out_prb_fld
        run_str = spec[0] + '_g' + spec[1]
        run_folder = 'catgt_' + run_str
        prb_folder = run_str + '_imec' + prb
        data_directory.append(os.path.join(catGT_dest, run_folder, prb_folder))
        fileName = run_str + '_tcat.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(data_directory[i], fileName)

        outputName = 'imec' + prb + '_ks2'

        # kilosort_postprocessing and noise_templates moduules alter the files
        # that are input to phy. If using these modules, keep a copy of the
        # original phy output
        if ('kilosort_postprocessing' in modules) or ('noise_templates' in modules):
            ks_make_copy = True
        else:
            ks_make_copy = False

        kilosort_output_dir = os.path.join(data_directory[i], outputName)

        print(data_directory[i])
        print(continuous_file)

        # get region specific parameters
        ks_Th = ksTh_dict.get(spec[4][i])
        refPerMS = refPerMS_dict.get(spec[4][i])

        info = createInputJson(module_input_json[i], npx_directory=npx_directory,
                               continuous_file=continuous_file,
                               spikeGLX_data=True,
                               input_meta_path=input_meta_fullpath,
                               kilosort_output_directory=kilosort_output_dir,
                               ks_make_copy=ks_make_copy,
                               noise_template_use_rf=False,
                               catGT_run_name=session_id[i],
                               gate_string=spec[1],
                               probe_string=spec[3],
                               ks_remDup=ks_remDup,
                               ks_finalSplits=1,
                               ks_labelGood=1,
                               ks_saveRez=ks_saveRez,
                               ks_copy_fproc=ks_copy_fproc,
                               ks_minfr_goodchannels=ks_minfr_goodchannels,
                               ks_whiteningRadius_um=ks_whiteningRadius_um,
                               ks_Th=ks_Th,
                               ks_CSBseed=1,
                               ks_LTseed=1,
                               ks_templateRadius_um=ks_templateRadius_um,
                               extracted_data_directory=catGT_dest,
                               event_ex_param_str=event_ex_param_str,
                               c_Waves_snr_um=c_Waves_snr_um,
                               qm_isi_thresh=refPerMS / 1000
                               )

        # copy json file to data directory as record of the input parameters

    # loop over probes for processing.
    for i, prb in enumerate(prb_list):
        run_one_probe.runOne(session_id[i],
                             json_directory,
                             data_directory[i],
                             run_CatGT,
                             catGT_input_json[i],
                             catGT_output_json[i],
                             modules,
                             module_input_json[i],
                             logFullPath)

    if runTPrime:
        # after loop over probes, run TPrime to create files of
        # event times -- edges detected in auxialliary files and spike times
        # from each probe -- all aligned to a reference stream.

        # create json files for calling TPrime
        session_id = spec[0] + '_TPrime'
        input_json = os.path.join(json_directory, session_id + '-input.json')
        output_json = os.path.join(json_directory, session_id + '-output.json')

        # build list of sync extractions to send to TPrime
        im_ex_list = ''
        for i, prb in enumerate(prb_list):
            sync_extract = '-SY=' + prb + ',-1,6,500'
            im_ex_list = im_ex_list + ' ' + sync_extract

        print('im_ex_list: ' + im_ex_list)

        info = createInputJson(input_json, npx_directory=npx_directory,
                               continuous_file=continuous_file,
                               spikeGLX_data=True,
                               input_meta_path=input_meta_fullpath,
                               catGT_run_name=spec[0],
                               kilosort_output_directory=kilosort_output_dir,
                               extracted_data_directory=catGT_dest,
                               tPrime_im_ex_list=im_ex_list,
                               tPrime_ni_ex_list=ni_extract_string,
                               event_ex_param_str=event_ex_param_str,
                               sync_period=1.0,
                               toStream_sync_params=toStream_sync_params,
                               niStream_sync_params=niStream_sync_params,
                               tPrime_3A=False,
                               toStream_path_3A=' ',
                               fromStream_list_3A=list()
                               )

        command = "python -W ignore -m ecephys_spike_sorting.modules." + 'tPrime_helper' + " --input_json " + input_json \
                  + " --output_json " + output_json
        subprocess.check_call(command.split(' '))


