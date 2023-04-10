import os
import sys

sys.path.append('/home/ryan/dev/SABer')
sys.path.append('/home/ryan/dev/SABer/src')
sys.path.append('/home/ryan/dev/SABer/src/saber')
import vamb_errstat
import saber_errstat
import unitem_errstat
import saber_LR_errstat

# Input files directory
working_dir = sys.argv[1]
synthdata_dir = sys.argv[2]
mg_asm = sys.argv[3]
sample_type = sys.argv[4]
sample = sys.argv[5]
mode = sys.argv[6]
params = sys.argv[7]
binner = sys.argv[8]
threads = int(sys.argv[9])

# SABer errstat
if binner == 'SABer':
    mockpath = os.path.join(synthdata_dir, 'Final_SAGs_20k_test_subset/' + str(sample) + '/')
    run_err_df = saber_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm,
                                                mockpath, sample_type, sample, mode, params,
                                                threads)
    run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
    print(binner)
    print(run_err_df)

if binner == 'VAMB':
    # VAMB errstat
    run_err_df = vamb_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm, sample_type, threads)
    run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
    print(binner)
    print(run_err_df)

if binner == 'UNITEM':
    # UNITEM errstat
    run_err_df = unitem_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm, sample_type, sample, mode, threads)
    run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
    print(binner)
    print(run_err_df)

if binner == 'SABer_LR':
    mockpath = os.path.join(synthdata_dir, 'Final_SAGs_20k_test_subset/' + str(sample) + '/')
    run_err_df = saber_LR_errstat.runErrorAnalysis(working_dir, synthdata_dir, mg_asm,
                                                mockpath, sample_type, sample, mode, params,
                                                threads)
    run_err_df.to_csv(os.path.join(working_dir, 'Bin.errstat.tsv'), sep='\t', index=False)
    print(binner)
    print(run_err_df)
