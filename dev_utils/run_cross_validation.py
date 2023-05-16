import glob
import itertools
import os
import sys

import pandas as pd

sys.path.append('/home/ryan/dev/SABer')
sys.path.append('/home/ryan/dev/SABer/src')
sys.path.append('/home/ryan/dev/SABer/src/saber')
import src.saber.abundance_recruiter as abr
import src.saber.clusterer as clst
import src.saber.minhash_recruiter as mhr
import src.saber.tetranuc_recruiter as tra
import saber_errstat as err

# This script is for running CV for HDBSCAN and OCSVM
# Expects that you have already pre-run the samples to create the
# coverage, tetra matrices, and feature embeddings (this was to save time)

# Params to test
nu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 'scale']
min_cluster_size = [5, 10, 15, 25, 50, 75, 100, 125, 150, 200]
min_samples = [5, 10, 15, 25, 40, 50, 75, 100, 125]
ocsvm_combo = list(itertools.product(*[nu, gamma]))
hdbscan_combo = list(itertools.product(*[min_cluster_size, min_samples]))

# Input files directory
working_dir = sys.argv[1]
synthdata_dir = sys.argv[2]
mg_asm = sys.argv[3]
sample = sys.argv[4]
threads = int(sys.argv[5])

sample_type = working_dir.split('/')[-4].rsplit('_')[-1]
sample_id = sample

# Find previously run files and build needed inputs
mhr_recruits = glob.glob(os.path.join(working_dir, '*.201.mhr_contig_recruits.tsv'))
if mhr_recruits:
    mhr_file = os.path.basename(mhr_recruits[0])
    mg_file = [mhr_file.replace('.201.mhr_contig_recruits.tsv', ''), None]
    ## Run MinHash recruiting algorithm
    #minhash_df_dict = mhr.run_minhash_recruiter(working_dir,
    #                                            working_dir,
    #                                            None, mg_file,
    #                                            threads
    #                                            )
else:
    minhash_df_dict = False

abr_recruits = glob.glob(os.path.join(working_dir, '*.coverage.scaled.tsv'))
abr_file = os.path.basename(abr_recruits[0])
mg_file = [abr_file.replace('.coverage.scaled.tsv', ''), None]
# Build abundance tables
#abund_file = abr.runAbundRecruiter(working_dir,
#                                   working_dir, mg_file,
#                                   None,
#                                   threads
#                                   )
# Build tetra hz tables
#tetra_file = tra.run_tetra_recruiter(working_dir,
#                                     mg_file
#                                     )

# Run iterative clusterings using all the different params from above
# First do HDBSCAN and keep OCSVM static
cat_err_list = []
mockpath = os.path.join(synthdata_dir, 'Final_SAGs_20k_rep1_' + str(sample) + '/')
for mcs, mss in hdbscan_combo:
    if mcs >= mss:
        print("Running HDBSCAN with min_cluster_size=" + str(mcs) + " and min_samples=" + str(mss))
        mg_id = mg_file[0]
        output_path = os.path.join(working_dir, '_'.join(['hdbscan', str(mcs), str(mss)]))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #clusters = clst.runClusterer(mg_id, working_dir, output_path, abund_file, tetra_file,
        #                             minhash_df_dict, mcs, mss, mcs, mss, 0.5, 'scale', threads
        #                             )
        mode = 'hdbscan_cv'
        params = '_'.join([str(x) for x in [mcs, mss, mcs, mss, 0.5, 'scale']])
        run_err_df = err.runErrorAnalysis(output_path, synthdata_dir, mg_asm, mockpath,
                                          sample_type, sample_id, mode, params, threads
                                          )
        cat_err_list.append(run_err_df)

# Next do the OCSVM
for nu, gamma in ocsvm_combo:
    print("Running OCSVM with nu=" + str(nu) + " and gamma=" + str(gamma))
    mg_id = mg_file[0]
    output_path = os.path.join(working_dir, '_'.join(['ocsvm', str(nu), str(gamma)]))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #clusters = clst.runClusterer(mg_id, working_dir, output_path, abund_file, tetra_file,
    #                             minhash_df_dict, 100, 25, 100, 25, nu, gamma, threads
    #                             )
    mode = 'ocsvm_cv'
    params = '_'.join([str(x) for x in [100, 25, 100, 25, nu, gamma]])
    run_err_df = err.runErrorAnalysis(output_path, synthdata_dir, mg_asm, mockpath,
                                      sample_type, sample_id, mode, params, threads
                                      )
    cat_err_list.append(run_err_df)

cat_err_df = pd.concat(cat_err_list)
cat_err_df.to_csv(os.path.join(working_dir, 'CV_errstat_new.tsv'), sep='\t', index=False)
