import os
import shutil
import subprocess
import numpy as np

from helpers import SpikeGLX_utils
from helpers import log_from_json
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

# -----------
# Input data
# -----------
# Name for log file for this pipeline run. Log file will be saved in the
# output destination directory catGT_dest
logName = 'foraging_ephys_log.csv'

# Raw data directory = npx_directory
# run_specs = name, gate, trigger and probes to process
npx_directory = r'D:\Data\Ephys\HH09'    #!!!
# npx_directory = r'F:'

# Each run_spec is a list of 4 strings:
#   undecorated run name (no g/t specifier, the run field in CatGT)
#   gate index, as a string (e.g. '0')
#   triggers to process/concatenate, as a string e.g. '0,400', '0,0 for a single file
#           can replace first limit with 'start', last with 'end'; 'start,end'
#           will concatenate all trials in the probe folder
#   probes to process, as a string, e.g. '0', '0,3', '0:3'

run_specs = [			#!!!
                        # ['HH09_20210406', '0', 'start,end', '0'],
                        # ['HH09_20210416', '0', 'start,end', '0:1'],
                        # ['HH09_20210418', '0', 'start,end', '0:1'],
                        # ['HH09_20210419', '0', 'start,end', '0:1'],
                       #  ['HH09_S06_20210609', '0', 'start,end', '0:1'],
                        ['HH09_S08_20210612', '0', 'start,end', '0:1'],

]

# ------------------
# Output destination
# ------------------
# Set to an existing directory; all output will be written here.
# Output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin
# catGT_dest = r'E:\catGT\HH100'       #!!!
# catGT_dest = r'E:\catGT\HH101_noGFix'       #!!!
catGT_dest = r'F:\catGT\HH09'       #!!!

# ------------
# CatGT params
# ------------
run_CatGT = True  # set to False to sort/process previously processed data.  #!!!
# catGT streams to process, e.g. just '-ap' for ap band only, '-ap -ni' for
# ap plus ni aux inputs
catGT_stream_string = '-ap -ni -lf'

# CatGT command string includes all instructions for catGT operations
# Note 1: directory naming in this script requires -prb_fld and -out_prb_fld
# Note 2: this command line includes specification of edge extraction
# see CatGT readme for details https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/
# catGT_cmd_string = '-prb_fld -out_prb_fld -gbldmx -gfix=0,0.10,0.02 -SY=0,384,6,500 -XA=1,1,3,500 -XD=2,0,0 -XD=2,1,10 -XD=2,1,50 -XD=2,1,1200'
# catGT_cmd_string = '-prb_fld -out_prb_fld -gbldmx -gfix=100,0.10,0.04 -SY=0,384,6,500 -XA=1,1,3,500 -XD=2,0,0 -XD=2,1,10 -XD=2,1,50 -XD=2,1,1200'   #!!!  # Non gfix. HH100, HH101

#!!! Note the order of NI files: XA analogs, then XD digital words. So if I have four XA channels, the XD will be #5 (4 in zero-based). Therefor -XD = 4, ...
# All pulse duration has +/- 20% tolerance

n_XAs = 3  # Number of analog channels
ni_sync = f'XD={n_XAs},0,500'      # Sync channel in NIDQ
main_npx_sync = 'SY=0,384,6,500'   # Sync channel in the main NPX (to whom NI and all other NPX channels will be mapped, typically imec0)
other_npx_sync = '-SY=1,384,6,500'   # Other imec channels!!!

catGT_cmd_string = ('-prb_fld -out_prb_fld '   # Mandatory
                    '-aphipass=300 -aplopass=9000 '   # Filters (from Dave)
                    #'-lfhipass=0.1 -lflopass=300 '
                    '-gbldmx -gfix=0.4,0.10,0.02 '    # peak >={a} mV, rising speed >={b}mV/sample-tick, back threshold {c} mV (from Dave). https://billkarsh.github.io/SpikeGLX/help/dmx_vs_gbl/dmx_vs_gbl/
                    f'-{main_npx_sync} '  # Sync pulse in imec file: probe #{a=imec0}, channel# {b=last channel in AP file, ie #384), #bit {c=#6 for 3B probe), pulse width {d=500} ms
                    f'-{ni_sync} '    # Sync pulse in nidq file: word {a=1 in Dave's rig now}, threshold {b=1}V, min {c=3}V, pulse width {d=500} ms
                    f'-XD={n_XAs},2,1 '   # Dig marker (actual bit code): 1 ms width
                    f'-XD={n_XAs},2,1.5 '   # Dig marker (start of bit code): word 4, bit 1, 1.5 ms
                    f'-XD={n_XAs},2,2 '   # Dig marker (Choice_L): 2 ms width
                    f'-XD={n_XAs},2,2.5 '   # Dig marker (Choice_R): 2.5 ms
                    f'-XD={n_XAs},2,10 '   # Dig marker (go cue): 10 ms width
                    f'-XD={n_XAs},2,20 '   # Dig marker (reward): 20 ms width
                    f'-XD={n_XAs},2,30 '   # Dig marker (ITI start): 30 ms
                    f'{other_npx_sync} '
                    )

# ----------------------
# psth_events parameters
# ----------------------
# extract param string for psth events -- copy the CatGT params used to extract
# events that should be exported with the phy output for PSTH plots
# If not using, remove psth_events from the list of modules
# Note that there must be no dash '-'!!!
# event_ex_param_str = 'XD=2,1,1200'  # HH100, HH101, no laser power
event_ex_param_str = f'XD={n_XAs},2,10'  # Go cue

# -----------------
# TPrime parameters
# -----------------
runTPrime = True   # set to False if not using TPrime
sync_period = 1.0   # true for SYNC wave generated by imec basestation

# Map all times to probe-0 (including event markers and spike times!!)
toStream_sync_params =  main_npx_sync  # copy from the CatGT command line, no spaces
niStream_sync_params =  ni_sync   # copy from the CatGT comman line, set to None if no Aux data, no spaces

# ---------------
# Modules List
# ---------------
# List of modules to run per probe; CatGT and TPrime are called once for each run.
modules = [    #!!!
            # 'depth_estimation',
 			'kilosort_helper',              # Run Kilosort 2
            'kilosort_postprocessing',        # Duplicate spike removal
            'noise_templates',                # Noise cluster ID
            'psth_events',                    # PSTH events for phy event_view plugin
            'mean_waveforms',                 #
            'quality_metrics'
			]

json_directory = r'F:\json_file'

# -----------------------
# -----------------------
# End of user input
# -----------------------
# -----------------------

# delete the existing CatGT.log
if run_CatGT:
    try:
        os.remove('CatGT.log')
    except OSError:
        pass

# delete existing Tprime.log
if runTPrime:
    try:
        os.remove('Tprime.log')
    except OSError:
        pass

# delete existing C_waves.log
try:
    os.remove('C_Waves.log')
except OSError:
    pass

# delete any existing log with the current name
logFullPath = os.path.join(catGT_dest, logName)
try:
    os.remove(logFullPath)
except OSError:
    pass

# create the log file, write header
log_from_json.writeHeader(logFullPath)

for spec in run_specs:

    session_id = spec[0]

    # Run CatGT
    input_json = os.path.join(json_directory, session_id + '-input.json')
    output_json = os.path.join(json_directory, session_id + '-output.json')

    # Make list of probes from the probe string
    prb_list = SpikeGLX_utils.ParseProbeStr(spec[3])

    # build path to the first probe folder
    run_folder_name = spec[0] + '_g' + spec[1]
    prb0_fld_name = run_folder_name + '_imec' + prb_list[0]
    prb0_fld = os.path.join(npx_directory, run_folder_name, prb0_fld_name)
    first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(spec[2], prb0_fld)
    trigger_str = repr(first_trig) + ',' + repr(last_trig)

    print('\n===================================\n\nCreating json file for preprocessing')
    info = createInputJson(input_json, npx_directory=npx_directory,
	                                   continuous_file = None,
                                       spikeGLX_data = 'True',
									   kilosort_output_directory=catGT_dest,
                                       catGT_run_name = session_id,
                                       gate_string = spec[1],
                                       trigger_string = trigger_str,
                                       probe_string = spec[3],
                                       catGT_stream_string = catGT_stream_string,
                                       catGT_cmd_string = catGT_cmd_string,
                                       extracted_data_directory = catGT_dest
                                       )

    # CatGT operates on whole runs with multiple probes, so gets called in just
    # once per run_spec
    if run_CatGT:
        command = "python -W ignore -m ecephys_spike_sorting.modules." + 'catGT_helper' + " --input_json " + input_json \
		          + " --output_json " + output_json
        subprocess.check_call(command.split(' '))

        # parse the CatGT log and write results to command line
        logPath = os.getcwd()
        gfix_edits = SpikeGLX_utils.ParseCatGTLog( logPath, spec[0], spec[1], prb_list )

        for i in range(0,len(prb_list)):
            edit_string = '{:.3f}'.format(gfix_edits[i])
            print('Probe ' + prb_list[i] + '; gfix edits/sec: ' + repr(gfix_edits[i]))
    else:
        # fill in dummy gfix_edits for running without preprocessing
        gfix_edits = np.zeros(len(prb_list), dtype='float64' )

    # finsihed preprocessing. All other modules are are called once per probe

    for i, prb in enumerate(prb_list):
        #create json files specific to this probe
        session_id = spec[0] + '_imec' + prb
        input_json = os.path.join(json_directory, session_id + '-input.json')


        # location of the binary created by CatGT, using -out_prb_fld
        run_str = spec[0] + '_g' + spec[1]
        run_folder = 'catgt_' + run_str
        prb_folder = run_str + '_imec' + prb
        data_directory = os.path.join(catGT_dest, run_folder, prb_folder)
        fileName = run_str + '_tcat.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(data_directory, fileName)

        outputName = 'imec' + prb + '_ks2'

        # kilosort_postprocessing and noise_templates moduules alter the files
        # that are input to phy. If using these modules, keep a copy of the
        # original phy output
        if ('kilosort_postprocessing' in modules) or('noise_templates' in modules):
            ks_make_copy = True
        else:
            ks_make_copy = False

        kilosort_output_dir = os.path.join(data_directory, outputName)

        print(data_directory)
        print(continuous_file)

        info = createInputJson(input_json, npx_directory=npx_directory,
	                                   continuous_file = continuous_file,
                                       spikeGLX_data = True,
									   kilosort_output_directory=kilosort_output_dir,
                                       ks_make_copy = ks_make_copy,
                                       noise_template_use_rf = False,
                                       catGT_run_name = session_id,
                                       gate_string = spec[1],
                                       trigger_string = trigger_str,
                                       probe_string = spec[3],
                                       catGT_stream_string = catGT_stream_string,
                                       catGT_cmd_string = catGT_cmd_string,
                                       catGT_gfix_edits = gfix_edits[i],
                                       extracted_data_directory = catGT_dest,
                                       event_ex_param_str = event_ex_param_str
                                       )

        # copy json file to data directory as record of the input parameters (and gfix edit rates)
        shutil.copy(input_json, os.path.join(data_directory, session_id + '-input.json'))

        for module in modules:
            output_json = os.path.join(json_directory, session_id + '-' + module + '-output.json')
            command = "python -W ignore -m ecephys_spike_sorting.modules." + module + " --input_json " + input_json \
		          + " --output_json " + output_json
            subprocess.check_call(command.split(' '))

        log_from_json.addEntry(modules, json_directory, session_id, logFullPath)

    if runTPrime:
        # after loop over probes, run TPrime to create files of
        # event times -- edges detected in auxialliary files and spike times
        # from each probe -- all aligned to a reference stream.

        # create json files for calling TPrime
        session_id = spec[0] + '_TPrime'
        input_json = os.path.join(json_directory, session_id + '-input.json')
        output_json = os.path.join(json_directory, session_id + '-output.json')

        info = createInputJson(input_json, npx_directory=npx_directory,
    	                                   continuous_file = continuous_file,
                                           spikeGLX_data = True,
    									   kilosort_output_directory=kilosort_output_dir,
                                           ks_make_copy = ks_make_copy,
                                           noise_template_use_rf = False,
                                           catGT_run_name = spec[0],
                                           gate_string = spec[1],
                                           trigger_string = trigger_str,
                                           probe_string = spec[3],
                                           catGT_stream_string = catGT_stream_string,
                                           catGT_cmd_string = catGT_cmd_string,
                                           catGT_gfix_edits = gfix_edits[i],
                                           extracted_data_directory = catGT_dest,
                                           event_ex_param_str = event_ex_param_str,
                                           sync_period = 1.0,
                                           toStream_sync_params = toStream_sync_params,
                                           niStream_sync_params = niStream_sync_params,
                                           toStream_path_3A = ' ',
                                           fromStream_list_3A = list()
                                           )

        command = "python -W ignore -m ecephys_spike_sorting.modules." + 'tPrime_helper' + " --input_json " + input_json \
    		          + " --output_json " + output_json
        subprocess.check_call(command.split(' '))
