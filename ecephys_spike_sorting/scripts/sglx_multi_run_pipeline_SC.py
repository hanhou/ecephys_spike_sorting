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
ksTh_dict = {'default':'[10,4]', 'cortex':'[10,4]', 'striatum':'[10,4]','medulla':'[10,4]','midbrain':'[10,4]', 'thalamus':'[10,4]'}
refPerMS_dict = {'default': 2.0, 'cortex': 2.0, 'striatum': 2.0, 'medulla': 1.0, 'midbrain': 1.5, 'thalamus': 1.5}

# -----------
# Input data
# -----------
# Name for log file for this pipeline run. Log file will be saved in the
# output destination directory catGT_dest
# If this file exists, new run data is appended to it
logName = 'SC064_log.csv'

# Raw data directory = npx_directory
# run_specs = name, gate, trigger and probes to process
npx_directory = r'G:'

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
#						['SC024_092319_NP1.0_Midbrain', '0', '0,9', '0,1', ('cortex', 'medulla') ],

                        # ['SC045_120920', '0', '0,0', '0,2,3', ['cortex','striatum','striatum'] ]
                        # ['SC045_121020', '0', '0,0', '0:3', ['cortex','cortex','striatum','striatum'] ],
                        # ['SC045_121120', '0', '0,0', '0:4', ['cortex','cortex','striatum','striatum','striatum'] ]
                        # ['SC045_121320', '0', '0,0', '0:2', ['cortex','midbrain','cortex'] ],
                        # ['SC045_121420', '0', '0,0', '0:3', ['cortex','cortex','midbrain','midbrain'] ],
                        # ['SC045_121520', '0', '0,0', '0:2', ['cortex','cortex','midbrain'] ],
                        # ['SC045_121620', '0', '0,0', '0:2', ['cortex','cortex','midbrain'] ],
                        # ['SC045_121720', '0', '0,0', '0:2', ['thalamus','thalamus','midbrain'] ],
                        # ['SC045_121820', '0', '0,0', '0:2', ['thalamus','thalamus','midbrain'] ]

                        # ['SC048_122420', '0', '0,0', '0:2', ['cortex','midbrain','thalamus'] ],
                        # ['SC048_122520', '0', '0,0', '0:2', ['cortex','midbrain','thalamus'] ],
                        # ['SC048_122620', '0', '0,0', '0:2', ['cortex','midbrain','thalamus'] ],
                        # ['SC048_122920', '0', '0,0', '0:3', ['cortex','cortex','midbrain','thalamus'] ],
                        # ['SC048_123020', '0', '0,0', '0:4', ['cortex','cortex','midbrain','thalamus','thalamus'] ],
                        # ['SC048_123120', '0', '0,0', '0:2', ['striatum','striatum','midbrain'] ],
                        # ['SC048_010121', '0', '0,0', '0:2', ['striatum','striatum','midbrain'] ],
                        # ['SC048_010321', '0', '0,0', '0:3', ['cortex','cortex','midbrain','midbrain'] ],
                        # ['SC048_010421', '0', '0,0', '0:3', ['cortex','cortex','midbrain','midbrain'] ]

                        # ['SC049_010621', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC049_010721', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC049_010821', '0', '0,0', '0:3', ['cortex','striatum','midbrain','striatum'] ],
                        # ['SC049_011021', '0', '0,0', '0:3', ['cortex','cortex','cortex','cortex'] ],
                        # ['SC049_011121', '0', '0,0', '0:3', ['cortex','cortex','cortex','cortex'] ],
                        # ['SC049_011221', '0', '0,0', '0:3', ['cortex','thalamus','thalamus','thalamus'] ],
                        # ['SC049_011321', '0', '0,0', '0:4', ['cortex','cortex','thalamus','thalamus','thalamus'] ]

                        # ['SC050_022521', '0', '0,0', '0:3', ['cortex','cortex','thalamus','thalamus'] ],
                        # ['SC050_022621', '0', '0,0', '0:3', ['cortex','cortex','thalamus','thalamus'] ],
                        # ['SC050_022721', '0', '0,0', '0:3', ['cortex','cortex','thalamus','thalamus'] ],
                        # ['SC050_030121', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC050_030221', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC050_030321', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC050_030421', '0', '0,0', '0:2', ['cortex','striatum','midbrain'] ]


                        # ['SC052_012121', '0', '0,0', '0:3', ['striatum','striatum','midbrain','midbrain'] ],
                        # ['SC052_012221', '0', '0,0', '0:3', ['striatum','striatum','midbrain','midbrain'] ],
                        # ['SC052_012321', '0', '0,0', '0:3', ['striatum','striatum','midbrain','midbrain'] ],
                        # ['SC052_012521', '0', '0,0', '0:4', ['cortex','cortex','midbrain','thalamus','thalamus'] ],
                        # ['SC052_012621', '0', '0,0', '0:3', ['cortex','midbrain','thalamus','thalamus'] ],
                        # ['SC052_012721', '0', '0,0', '0:4', ['cortex','cortex','midbrain','thalamus','thalamus'] ],
                        # ['SC052_012821', '0', '0,0', '0:4', ['cortex','cortex','midbrain','thalamus','thalamus'] ],
                        # ['SC052_012921', '0', '0,0', '0:4', ['cortex','cortex','thalamus','thalamus','thalamus'] ]
                        # ['SC052_013021', '0', '0,0', '0:3', ['cortex','cortex','cortex','cortex'] ]

                        # ['SC053_021821', '0', '0,0', '0:3', ['cortex','thalamus','striatum','midbrain'] ],
                        # ['SC053_021921', '0', '0,0', '0:2', ['cortex','thalamus','cortex'] ],
                        # ['SC053_022021', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ],
                        # ['SC053_022121', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ],
                        # ['SC053_022221', '0', '0,0', '0:2', ['cortex','cortex','midbrain'] ],
                        # ['SC053_022321', '0', '0,0', '0:2', ['thalamus','cortex','midbrain'] ],
                        # ['SC053_022421', '0', '0,0', '0:3', ['thalamus','cortex','midbrain','striatum'] ],
                        # ['SC053_022521', '0', '0,0', '0:3', ['thalamus','cortex','midbrain','striatum'] ],
                        # ['SC053_022621', '0', '0,0', '0:3', ['thalamus','cortex','midbrain','striatum'] ],
                        # ['SC053_022821', '0', '0,0', '0:2', ['striatum','midbrain','striatum'] ]

                        # ['SC060_031821', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ]
                        # ['SC060_031921', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ],
                        # ['SC060_032021', '0', '0,0', '0:3', ['cortex','cortex','midbrain','thalamus'] ],
                        # ['SC060_032221', '0', '0,0', '0:3', ['cortex','cortex','striatum','midbrain'] ],
                        # ['SC060_032321', '0', '0,0', '0:3', ['cortex','cortex','striatum','midbrain'] ],
                        # ['SC060_032421', '0', '0,0', '0:2', ['cortex','cortex','thalamus'] ]

                        # ['SC061_031821', '0', '0,0', '0:3', ['cortex','cortex','midbrain','striatum'] ]
                        # ['SC061_031921', '0', '0,0', '0:3', ['cortex','midbrain','striatum','striatum'] ],
                        # ['SC061_032121', '0', '0,0', '0:3', ['cortex','cortex','striatum','midbrain'] ],
                        # ['SC061_032221', '0', '0,0', '0:3', ['cortex','cortex','striatum','midbrain'] ],
                        # ['SC061_032321', '0', '0,0', '0:3', ['striatum','cortex','thalamus','thalamus'] ],
                        # ['SC061_032421', '0', '0,0', '0:2', ['striatum','thalamus','thalamus'] ]

                        ['SC064_042721', '0', '0,0', '0:4', ['cortex','thalamus','cortex','midbrain','thalamus'] ],
                        ['SC064_042821', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ],
                        ['SC064_042921', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ]
                        # ['SC064_043021', '0', '0,0', '0:4', ['cortex','thalamus','cortex','midbrain','thalamus'] ]
                        # ['SC064_050421', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC064_050521', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC064_050621', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC064_050721', '0', '0,0', '0:4', ['striatum','cortex','thalamus','cortex','thalamus'] ]
                        # ['SC064_050821', '0', '0,0', '0:3', ['striatum','cortex','thalamus','cortex'] ]

                        # ['SC065_050421', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC065_050521', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC065_050621', '0', '0,0', '0:3', ['striatum','cortex','midbrain','cortex'] ],
                        # ['SC065_050721', '0', '0,0', '0:1', ['striatum','thalamus'] ]
                        # ['SC065_050921', '0', '0,0', '0:3', ['striatum','cortex','thalamus','cortex'] ],
                        # ['SC065_051021', '0', '0,0', '0:3', ['cortex','striatum','cortex','midbrain'] ],
                        # ['SC065_051121', '0', '0,0', '0:3', ['cortex','striatum','cortex','midbrain'] ],
                        # ['SC065_051321', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ],
                        # ['SC065_051421', '0', '0,0', '0:3', ['cortex','thalamus','cortex','midbrain'] ]

                        # ['SC066_041321', '0', '0,0', '0:3', ['cortex','striatum','midbrain','medulla'] ],
                        # ['SC066_041421', '0', '0,0', '0:3', ['cortex','striatum','midbrain','medulla'] ],
                        # ['SC066_041521', '0', '0,0', '0:3', ['cortex','striatum','midbrain','medulla'] ],
                        # ['SC066_041621', '0', '0,0', '0:3', ['cortex','striatum','midbrain','medulla'] ]
                        # ['SC066_041921', '0', '0,0', '0:4', ['striatum','cortex','medulla','midbrain','medulla'] ],
                        # ['SC066_042021', '0', '0,0', '0:4', ['striatum','cortex','medulla','midbrain','medulla'] ],
                        # ['SC066_042121', '0', '0,0', '0:3', ['striatum','cortex','medulla','midbrain'] ]

                        # ['SC067_041321', '0', '0,0', '0:4', ['cortex','striatum','midbrain','medulla','medulla'] ],
                        # ['SC067_041421', '0', '0,0', '0:4', ['cortex','striatum','midbrain','medulla','medulla'] ],
                        # ['SC067_041521', '0', '0,0', '0:3', ['cortex','striatum','midbrain','medulla'] ],
                        # ['SC067_041821', '0', '0,0', '0:4', ['striatum','cortex','medulla','midbrain','medulla'] ]
                        # ['SC067_041921', '0', '0,0', '0:4', ['striatum','cortex','medulla','midbrain','medulla'] ],
                        # ['SC067_042021', '0', '0,0', '0:4', ['striatum','cortex','medulla','midbrain','medulla'] ]

]

# ------------------
# Output destination
# ------------------
# Set to an existing directory; all output will be written here.
# Output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin
catGT_dest = r'Z:\Ingested\SC064'

# ------------
# CatGT params
# ------------
run_CatGT = True   # set to False to sort/process previously processed data.


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

ni_present = True
ni_extract_string = '-XA=0,1,3,500 -XA=1,3,3,0 -iXA=4,2.5,1,0 -iXA=5,2.5,1,0 -iXA=6,2.5,1,0 -XD=7,1,50 -XD=7,2,1.7 -XD=7,3,5'


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
event_ex_param_str = 'XD=7,1,50'

# -----------------
# TPrime parameters
# -----------------
runTPrime = True   # set to False if not using TPrime
sync_period = 1.0   # true for SYNC wave generated by imec basestation
toStream_sync_params = 'SY=0,-1,6,500'  # copy from the CatGT command line, no spaces
niStream_sync_params = 'XA=0,1,3,500'   # copy from the CatGT comman line, set to None if no Aux data, no spaces

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

json_directory = r'C:\Users\scanimage\Documents\ecephys_clone\json_files'

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

        #create CatGT command for this probe
        print('Creating json file for CatGT on probe: ' + prb)
        # Run CatGT
        catGT_input_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-input.json'))
        catGT_output_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-output.json'))

        # build extract string for SYNC channel for this probe
        sync_extract = '-SY=' + prb +',-1,6,500'

        # if this is the first probe proceessed, process the ni stream with it
        if i == 0 and ni_present:
            catGT_stream_string = '-ap -ni'
            extract_string = sync_extract + ' ' + ni_extract_string
        else:
            catGT_stream_string = '-ap'
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
                                       continuous_file = continuous_file,
                                       kilosort_output_directory=catGT_dest,
                                       spikeGLX_data = True,
                                       input_meta_path = input_meta_fullpath,
                                       catGT_run_name = spec[0],
                                       gate_string = spec[1],
                                       trigger_string = trigger_str,
                                       probe_string = prb,
                                       catGT_stream_string = catGT_stream_string,
                                       catGT_car_mode = car_mode,
                                       catGT_loccar_min_um = loccar_min,
                                       catGT_loccar_max_um = loccar_max,
                                       catGT_cmd_string = catGT_cmd_string + ' ' + extract_string,
                                       extracted_data_directory = catGT_dest
                                       )

        #create json files for the other modules
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
        if ('kilosort_postprocessing' in modules) or('noise_templates' in modules):
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
	                                   continuous_file = continuous_file,
                                       spikeGLX_data = True,
                                       input_meta_path = input_meta_fullpath,
									   kilosort_output_directory=kilosort_output_dir,
                                       ks_make_copy = ks_make_copy,
                                       noise_template_use_rf = False,
                                       catGT_run_name = session_id[i],
                                       gate_string = spec[1],
                                       probe_string = spec[3],
                                       ks_remDup = ks_remDup,
                                       ks_finalSplits = 1,
                                       ks_labelGood = 1,
                                       ks_saveRez = ks_saveRez,
                                       ks_copy_fproc = ks_copy_fproc,
                                       ks_minfr_goodchannels = ks_minfr_goodchannels,
                                       ks_whiteningRadius_um = ks_whiteningRadius_um,
                                       ks_Th = ks_Th,
                                       ks_CSBseed = 1,
                                       ks_LTseed = 1,
                                       ks_templateRadius_um = ks_templateRadius_um,
                                       extracted_data_directory = catGT_dest,
                                       event_ex_param_str = event_ex_param_str,
                                       c_Waves_snr_um = c_Waves_snr_um,
                                       qm_isi_thresh = refPerMS/1000
                                       )

        # copy json file to data directory as record of the input parameters


    # loop over probes for processing.
    for i, prb in enumerate(prb_list):

        run_one_probe.runOne( session_id[i],
                 json_directory,
                 data_directory[i],
                 run_CatGT,
                 catGT_input_json[i],
                 catGT_output_json[i],
                 modules,
                 module_input_json[i],
                 logFullPath )


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
            sync_extract = '-SY=' + prb +',-1,6,500'
            im_ex_list = im_ex_list + ' ' + sync_extract

        print('im_ex_list: ' + im_ex_list)

        info = createInputJson(input_json, npx_directory=npx_directory,
    	                                   continuous_file = continuous_file,
                                           spikeGLX_data = True,
                                           input_meta_path = input_meta_fullpath,
                                           catGT_run_name = spec[0],
    									   kilosort_output_directory=kilosort_output_dir,
                                           extracted_data_directory = catGT_dest,
                                           tPrime_im_ex_list = im_ex_list,
                                           tPrime_ni_ex_list = ni_extract_string,
                                           event_ex_param_str = event_ex_param_str,
                                           sync_period = 1.0,
                                           toStream_sync_params = toStream_sync_params,
                                           niStream_sync_params = niStream_sync_params,
                                           tPrime_3A = False,
                                           toStream_path_3A = ' ',
                                           fromStream_list_3A = list()
                                           )

        command = sys.executable + " -W ignore -m ecephys_spike_sorting.modules." + 'tPrime_helper' + " --input_json " + input_json \
    		          + " --output_json " + output_json
        subprocess.check_call(command.split(' '))


