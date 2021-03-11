import argparse
import logging
import multiprocessing
import warnings
from functools import reduce
from os.path import isfile, basename
from os.path import join as o_join

import numpy as np
import pandas as pd
import saber.logger as s_log
import saber.utilities as s_utils
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture as BayesGMM
from sklearn.mixture import GaussianMixture as GMM

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO: not really sure what else to do about this error:
# /home/rmclaughlin/anaconda3/envs/saber_env/lib/python3.8/site-packages/pandas/core/ops/array_ops.py:253:
# FutureWarning: elementwise comparison failed;
# returning scalar instead, but in the future will perform elementwise comparison
# res_values = method(rvalues)


def run_tetra_recruiter(tra_path, sag_sub_files, mg_sub_file, abund_recruit_df, minhash_df,
                        per_pass, nthreads, force
                        ):
    """Returns dataframe of subcontigs recruited via tetranucleotide Hz

    Parameters:
    tra_path (str): string of global path to tetrenucleotide output directory
    sag_sub_files (list): list containing sublists with two values: [sag_id, sag_path]
                          where sag_id (str) is a unique ID for a SAG and sag_path is
                          the global path the the SAG subcontig fasta file
    mg_sub_file (list): list containing two values: mg_id and mg_file. (same as sag_sub_files)
    rpkm_max_df (df): dataframe containing the abundance recruits from the previous step.
    per_pass (float): percent of agreement for subcontig classification to pass the complete
                          contig (default is 0.01)

    """
    # TODO: 1. Think about using Minimum Description Length (MDL) instead of AIC/BIC
    #        2. [Normalized Maximum Likelihood or Fish Information Approximation]
    #        3. Can TetraNuc Hz be calc'ed for each sample? Does that improve things?
    #            (think about http://merenlab.org/2020/01/02/visualizing-metagenomic-bins/#introduction)

    logging.info('Starting Tetranucleotide Recruitment\n')
    mg_id = mg_sub_file[0]
    # Build/Load tetramers for SAGs and MG subset by ara recruits
    if isfile(o_join(tra_path, mg_id + '.tetras.tsv')):
        logging.info('Loading tetramer Hz matrix for %s\n' % mg_id)
        mg_tetra_df = pd.read_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                                  sep='\t', index_col=0, header=0
                                  )
        mg_headers = mg_tetra_df.index.values
    else:
        logging.info('Calculating tetramer Hz matrix for %s\n' % mg_id)
        mg_subcontigs = s_utils.get_seqs(mg_sub_file[1])
        mg_headers = tuple(mg_subcontigs.keys())
        mg_subs = tuple([r.seq for r in mg_subcontigs])
        mg_tetra_df = s_utils.tetra_cnt(mg_subcontigs)
        mg_tetra_df.to_csv(o_join(tra_path, mg_id + '.tetras.tsv'),
                           sep='\t'
                           )
    contig_ids = list(zip(*mg_tetra_df.index.str.rsplit("_", n=1, expand=True).to_list()))[0]
    mg_tetra_df['contig_id'] = contig_ids
    mg_tetra_df.index.names = ['subcontig_id']
    std_tetra_dict = build_uniq_dict(mg_tetra_df, 'contig_id', nthreads, 'TetraHz')
    # Prep MinHash
    minhash_df.sort_values(by='jacc_sim', ascending=False, inplace=True)
    minhash_dedup_df = minhash_df[['sag_id', 'subcontig_id', 'contig_id', 'jacc_sim', 'jacc_sim_max']
    ].loc[minhash_df['jacc_sim_max'] == 1.0].drop_duplicates(subset=['sag_id', 'contig_id'])
    mh_recruit_dict = build_uniq_dict(minhash_dedup_df, 'sag_id', nthreads,
                                      'MinHash Recruits')  # TODO: this might not need multithreading
    # Prep Abundance
    # abund_dedup_df = abund_recruit_df[['sag_id', 'contig_id']].drop_duplicates(subset=['sag_id', 'contig_id'])
    ab_recruit_dict = build_uniq_dict(abund_recruit_df, 'sag_id', nthreads,
                                      'Abundance Recruits')  # TODO: this might not need multithreading
    # Subset tetras matrix for each SAG
    sag_id_list = list(mh_recruit_dict.keys())
    sag_id_cnt = len(sag_id_list)
    gmm_df_list = []
    svm_df_list = []
    iso_df_list = []
    comb_df_list = []
    sag_chunks = [list(x) for x in np.array_split(np.array(list(sag_id_list)), nthreads * 2)
                  if len(x) != 0
                  ]
    s_counter = 0
    r_counter = 0
    for i, sag_id_chunk in enumerate(sag_chunks, 1):
        pool = multiprocessing.Pool(processes=nthreads)
        arg_list = []
        for j, sag_id in enumerate(sag_id_chunk, 1):  # TODO: reduce RAM usage
            s_counter += 1
            logging.info('\rPrepping to Run TetraHz ML-Ensemble: Block {} - {}/{}'.format(i, s_counter, sag_id_cnt))
            if ((sag_id in list(mh_recruit_dict.keys())) & (sag_id in list(ab_recruit_dict.keys()))):
                minhash_sag_df = mh_recruit_dict.pop(sag_id)
                abund_sag_df = ab_recruit_dict.pop(sag_id)
                if ((minhash_sag_df.shape[0] != 0) & (abund_sag_df.shape[0] != 0)):
                    sag_id, sag_tetra_df, mg_tetra_filter_df = subset_tetras([std_tetra_dict,
                                                                              minhash_sag_df,
                                                                              abund_sag_df, sag_id
                                                                              ])
                    arg_list.append([force, mg_headers, mg_tetra_filter_df, sag_id, sag_tetra_df, tra_path])
        arg_list = tuple(arg_list)
        results = pool.imap_unordered(ensemble_recruiter, arg_list)
        logging.info('\n')
        for k, output in enumerate(results, 1):  # TODO: maybe check if files exist before running this, like minhash
            r_counter += 1
            logging.info('\rRecruiting with TetraHz ML-Ensemble: Block {} - {}/{}'.format(i, r_counter, sag_id_cnt))
            comb_recruits_df, gmm_recruits_df, iso_recruits_df, svm_recruits_df = output
            if isinstance(gmm_recruits_df, pd.DataFrame):
                gmm_df_list.append(gmm_recruits_df)
            if isinstance(svm_recruits_df, pd.DataFrame):
                svm_df_list.append(svm_recruits_df)
            if isinstance(iso_recruits_df, pd.DataFrame):
                iso_df_list.append(iso_recruits_df)
            if isinstance(comb_recruits_df, pd.DataFrame):
                comb_df_list.append(comb_recruits_df)
        logging.info('\n')
        pool.close()
        pool.join()
    gmm_concat_df = pd.concat(gmm_df_list)
    svm_concat_df = pd.concat(svm_df_list)
    iso_concat_df = pd.concat(iso_df_list)
    comb_concat_df = pd.concat(comb_df_list)
    gmm_concat_df.to_csv(o_join(tra_path, mg_id + '.gmm.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    svm_concat_df.to_csv(o_join(tra_path, mg_id + '.svm.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    iso_concat_df.to_csv(o_join(tra_path, mg_id + '.iso.tra_trimmed_recruits.tsv'), sep='\t',
                         index=False
                         )
    comb_concat_df.to_csv(o_join(tra_path, mg_id + '.comb.tra_trimmed_recruits.tsv'), sep='\t',
                          index=False
                          )

    tetra_df_dict = {'gmm': gmm_concat_df, 'svm': svm_concat_df, 'iso': iso_concat_df,
                     'comb': comb_concat_df
                     }

    return tetra_df_dict


def ensemble_recruiter(p):
    force, mg_headers, mg_tetra_filter_df, sag_id, sag_tetra_df, tra_path = p

    # Data reduction/de-noising with KMEANS clustering
    kmeans_pass_list = runKMEANS(sag_tetra_df, sag_id, mg_tetra_filter_df)
    kmeans_pass_df = pd.DataFrame(kmeans_pass_list,
                                  columns=['sag_id', 'subcontig_id', 'contig_id']
                                  )
    if kmeans_pass_df.shape[0] > 1:
        mg_tetra_kmeans_df = mg_tetra_filter_df.loc[mg_tetra_filter_df.index.isin(
            kmeans_pass_df['subcontig_id'])]
        # Recruit tetras with OC-SVM
        svm_recruits_df = OCSVM_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                          sag_tetra_df, tra_path, force
                                          )
        # Recruit tetras with GMM
        gmm_recruits_df = GMM_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                        sag_tetra_df, tra_path, force
                                        )
        # Recruit tetras with ISO-F
        iso_recruits_df = ISO_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                        sag_tetra_df, tra_path, force
                                        )
        '''
        # Recruit tetras with OC-SVM
        svm_recruits_df, svm_pred_df = OCSVM_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                                       sag_tetra_df, tra_path, force
                                                       )
        # Recruit tetras with GMM
        gmm_recruits_df, gmm_pred_df = GMM_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                                     sag_tetra_df, tra_path, force
                                                     )
        # Recruit tetras with ISO-F
        iso_recruits_df, iso_pred_df = ISO_recruiter(mg_headers, mg_tetra_kmeans_df, sag_id,
                                                     sag_tetra_df, tra_path, force
                                                     )
        pred_dfs = [svm_pred_df, gmm_pred_df, iso_pred_df]
        all_preds_df = reduce(lambda left, right: pd.merge(left, right,
                                                           left_index=True, right_index=True
                                                           ), pred_dfs)
        all_preds_df['total_pred'] = all_preds_df.sum(axis=1)
        passed_preds_df = all_preds_df.loc[all_preds_df['total_pred'] >= 2]
        '''
    else:
        svm_recruits_df = None
        gmm_recruits_df = None
        iso_recruits_df = None

    if isinstance(gmm_recruits_df, pd.DataFrame):
        '''
        # Try new method with all predictions
        comb_recruits_df = build_new_Ensemble(passed_preds_df, mg_headers, sag_id,
                                          tra_path, force
                                          )
        '''
        # Build Ensemble recruit DF, quality filter as well
        comb_recruits_df = build_Ensemble(gmm_recruits_df, svm_recruits_df, iso_recruits_df, mg_headers, sag_id,
                                          tra_path, force
                                          )
    else:
        comb_recruits_df = None

    return comb_recruits_df, gmm_recruits_df, iso_recruits_df, svm_recruits_df


def build_uniq_dict(src_df, col_val, nthreads, df_type):
    uniq_vals = list(src_df[col_val].unique())
    uniq_count = len(uniq_vals)
    uniq_chunk = [list(x) for x in np.array_split(np.array(list(uniq_vals)), nthreads) if len(x) != 0]
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for i, uniq_vals in enumerate(uniq_chunk, 1):
        arg_list.append([col_val, uniq_vals, src_df])
    results = pool.imap_unordered(build_df_dict, arg_list)
    uniq_dict = {}
    contig_count = 0
    logging.info('\rBuilding ' + col_val + ' Dictionary for ' + df_type + ': {}/{}'.format(contig_count, uniq_count))
    for i, out_dict in enumerate(results, 1):
        contig_count += len(out_dict.keys())
        logging.info(
            '\rBuilding ' + col_val + ' Dictionary for ' + df_type + ': {}/{}'.format(contig_count, uniq_count))
        uniq_dict.update(out_dict)
    logging.info('\n')
    pool.close()
    pool.join()
    return uniq_dict


def build_df_dict(p):
    col_str, uniq_list, temp_df = p
    uniq_dict = {}
    for i, val in enumerate(uniq_list):
        uniq_dict[val] = temp_df.loc[temp_df[col_str] == val]
    return uniq_dict


def build_new_Ensemble(passed_preds_df, mg_headers, sag_id, tra_path, force):
    passed_preds_df.reset_index(inplace=True)
    passed_preds_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
    passed_preds_df['sag_id'] = sag_id
    passed_preds_df['contig_id'] = [x.rsplit('_', 1)[0] for x in passed_preds_df['subcontig_id']]
    cnt_df = passed_preds_df.groupby(['sag_id', 'contig_id']
                                     ).count().reset_index().set_index(['sag_id', 'contig_id'])
    del cnt_df['subcontig_id']
    del cnt_df['total_pred']
    sum_df = passed_preds_df.groupby(['sag_id', 'contig_id']
                                     ).sum().reset_index().set_index(['sag_id', 'contig_id'])
    del sum_df['total_pred']
    percent_df = sum_df / cnt_df
    percent_df['total_pred'] = percent_df.sum(axis=1)
    percent_df.reset_index(inplace=True)
    filtered_preds_df = percent_df.loc[percent_df['total_pred'] > 1.0]
    filtered_subcontigs_df = passed_preds_df.loc[
        passed_preds_df['contig_id'].isin(filtered_preds_df['contig_id'])
    ]
    clean_pred_df = pd.DataFrame(filtered_subcontigs_df['subcontig_id'],
                                 columns=['subcontig_id']
                                 )
    clean_pred_df['contig_id'] = [x.rsplit('_', 1)[0] for x in clean_pred_df['subcontig_id']]
    clean_pred_df['sag_id'] = sag_id
    comb_df = clean_pred_df[['sag_id', 'subcontig_id', 'contig_id']]
    comb_filter_df = filter_tetras(sag_id, mg_headers, 'comb', comb_df)
    comb_filter_df.to_csv(o_join(tra_path, sag_id + '.comb_recruits.tsv'),
                          sep='\t', index=False
                          )
    return comb_filter_df


def build_Ensemble(gmm_filter_df, svm_filter_df, iso_filter_df, mg_headers, sag_id, tra_path, force):
    if (isfile(o_join(tra_path, sag_id + '.comb_recruits.tsv')) &
            (force is False)
    ):
        # logging.info('\nLoading  %s tetramer Hz recruit list\n' % sag_id)
        comb_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.comb_recruits.tsv'),
                                     sep='\t', header=0
                                     )
    else:
        dfs = [gmm_filter_df, svm_filter_df, iso_filter_df]
        comb_recruit_df = reduce(lambda left, right: pd.merge(left, right,
                                                              on=['sag_id', 'subcontig_id', 'contig_id']
                                                              ), dfs
                                 )
        comb_recruit_df.dropna(subset=['gmm_p', 'svm_p', 'iso_p'], how='any',
                               inplace=True
                               )
        pscale_cols = [x for x in comb_recruit_df.columns if '_w' in x]
        comb_recruit_df['ensemble_score'] = comb_recruit_df[pscale_cols].sum(axis=1)
        '''
        gmm_id_list = list(gmm_filter_df['subcontig_id'])
        svm_id_list = list(svm_filter_df['subcontig_id'])
        iso_id_list = list(iso_filter_df['subcontig_id'])
        gmm_svm_set = set(gmm_id_list).intersection(svm_id_list)
        iso_svm_set = set(iso_id_list).intersection(svm_id_list)
        gmm_iso_set = set(gmm_id_list).intersection(iso_id_list)
        # comb_set_list = list(set(list(gmm_svm_set) + list(iso_svm_set)))
        # comb_set_list = list(set(list(gmm_svm_set) + list(iso_svm_set) + list(gmm_iso_set)))
        comb_set_list = list(set(svm_id_list).intersection(gmm_iso_set))
        comb_pass_list = []
        for md_nm in comb_set_list:
            comb_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
        comb_df = pd.DataFrame(comb_pass_list,
                               columns=['sag_id', 'subcontig_id', 'contig_id']
                               )
        comb_filter_df = filter_tetras(sag_id, mg_headers, 'comb', comb_df)
        '''
        comb_filter_df = comb_recruit_df.loc[comb_recruit_df['ensemble_score'] >= 0.1]
        comb_filter_df.to_csv(o_join(tra_path, sag_id + '.comb_recruits.tsv'),
                              sep='\t', index=False
                              )
    return comb_filter_df


def ISO_recruiter(mg_headers, mg_tetra_filter_df, sag_id, sag_tetra_df, tra_path, force):
    if (isfile(o_join(tra_path, sag_id + '.iso_recruits.tsv')) &
            (force is False)
    ):
        # logging.info('\nLoading  %s tetramer Hz recruit list\n' % sag_id)
        iso_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.iso_recruits.tsv'),
                                    sep='\t', header=0
                                    )
    else:
        # fit IsoForest
        clf = IsolationForest(n_estimators=1000, random_state=42)
        clf.fit(sag_tetra_df.values)
        sag_pred = clf.predict(sag_tetra_df.values)
        sag_score = clf.decision_function(sag_tetra_df.values)
        # TODO:
        #  convert sag_score and mg_score to normalized anomaly_score
        #  as defined by score_samples/decision_fuction
        sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_tetra_df.index.values,
                                   columns=['anomaly'])
        sag_pred_df.loc[sag_pred_df['anomaly'] == 1, 'anomaly'] = 0
        sag_pred_df.loc[sag_pred_df['anomaly'] == -1, 'anomaly'] = 1
        sag_pred_df['scores'] = sag_score
        lower_bound, upper_bound = iqr_bounds(sag_pred_df['scores'], k=0.5)

        mg_pred = clf.predict(mg_tetra_filter_df.values)
        mg_score = clf.decision_function(mg_tetra_filter_df.values)
        mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values,
                                  columns=['anomaly'])
        mg_pred_df.loc[mg_pred_df['anomaly'] == 1, 'anomaly'] = 0
        mg_pred_df.loc[mg_pred_df['anomaly'] == -1, 'anomaly'] = 1
        mg_pred_df['scores'] = mg_score
        mg_pred_df['iqr_anomaly'] = 0
        mg_pred_df['iqr_anomaly'] = (mg_pred_df['scores'] < lower_bound) | \
                                    (mg_pred_df['scores'] > upper_bound)
        mg_pred_df['iqr_anomaly'] = mg_pred_df['iqr_anomaly'].astype(int)
        iso_pass_df = mg_pred_df.loc[mg_pred_df['iqr_anomaly'] != 1]
        iso_pass_list = []
        for md_nm in iso_pass_df.index.values:
            iso_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
        iso_df = pd.DataFrame(iso_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
        iso_filter_df = filter_tetras(sag_id, mg_headers, 'iso', iso_df)
        iso_filter_df.to_csv(o_join(tra_path, sag_id + '.iso_recruits.tsv'),
                             sep='\t', index=False
                             )
        trimmed_mg_pred_df = mg_pred_df[['iqr_anomaly']]
        trimmed_mg_pred_df.rename(columns={'iqr_anomaly': 'iso_pred'}, inplace=True)
    return iso_filter_df  # , trimmed_mg_pred_df


def OCSVM_recruiter(mg_headers, mg_tetra_filter_df, sag_id, sag_tetra_df, tra_path, force):
    if (isfile(o_join(tra_path, sag_id + '.svm_recruits.tsv')) &
            (force is False)
    ):
        # logging.info('\nLoading  %s tetramer Hz recruit list\n' % sag_id)
        svm_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.svm_recruits.tsv'),
                                    sep='\t', header=0
                                    )
    else:
        # logging.info('Training OCSVM on SAG tetras\n')
        # fit OCSVM
        clf = svm.OneClassSVM(nu=0.9, gamma=0.0001)
        clf.fit(sag_tetra_df.values)
        # print(clf.get_params())
        sag_pred = clf.predict(sag_tetra_df.values)
        # sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_tetra_df.index.values)
        mg_pred = clf.predict(mg_tetra_filter_df.values)
        mg_pred_df = pd.DataFrame(data=mg_pred, index=mg_tetra_filter_df.index.values,
                                  columns=['ocsvm_pred']
                                  )
        svm_pass_df = mg_pred_df.loc[mg_pred_df['ocsvm_pred'] != -1]
        svm_pass_list = []
        for md_nm in svm_pass_df.index.values:
            svm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
        svm_df = pd.DataFrame(svm_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
        svm_filter_df = filter_tetras(sag_id, mg_headers, 'svm', svm_df)
        svm_filter_df.to_csv(o_join(tra_path, sag_id + '.svm_recruits.tsv'),
                             sep='\t', index=False
                             )
        mg_pred_df.replace(to_replace=-1, value=0, inplace=True)
    return svm_filter_df  # , mg_pred_df


def GMM_recruiter(mg_headers, mg_tetra_filter_df, sag_id, sag_tetra_df, tra_path, force):
    if (isfile(o_join(tra_path, sag_id + '.gmm_recruits.tsv')) &
            (force is False)
    ):
        # logging.info('\nLoading  %s tetramer Hz recruit list\n' % sag_id)
        gmm_filter_df = pd.read_csv(o_join(tra_path, sag_id + '.gmm_recruits.tsv'),
                                    sep='\t', header=0
                                    )
    else:
        gmm = BayesGMM(weight_concentration_prior_type='dirichlet_process', random_state=42
                       ).fit(sag_tetra_df.values)
        sag_score = gmm.score_samples(sag_tetra_df.values)
        sag_pred_df = pd.DataFrame(data=sag_score, index=sag_tetra_df.index.values,
                                   columns=['scores'])
        lower_bound, upper_bound = iqr_bounds(sag_pred_df['scores'], k=3.0)

        mg_score = gmm.score_samples(mg_tetra_filter_df.values)
        mg_pred_df = pd.DataFrame(data=mg_score, index=mg_tetra_filter_df.index.values,
                                  columns=['scores'])
        mg_pred_df['iqr_anomaly'] = 0
        mg_pred_df['iqr_anomaly'] = (mg_pred_df['scores'] < lower_bound) | \
                                    (mg_pred_df['scores'] > upper_bound)
        mg_pred_df['iqr_anomaly'] = mg_pred_df['iqr_anomaly'].astype(int)
        gmm_pass_df = mg_pred_df.loc[mg_pred_df['iqr_anomaly'] != 1]
        gmm_pass_list = []
        for md_nm in gmm_pass_df.index.values:
            gmm_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
        gmm_df = pd.DataFrame(gmm_pass_list, columns=['sag_id', 'subcontig_id', 'contig_id'])
        gmm_filter_df = filter_tetras(sag_id, mg_headers, 'gmm', gmm_df)
        gmm_filter_df.to_csv(o_join(tra_path, sag_id + '.gmm_recruits.tsv'),
                             sep='\t', index=False
                             )
        trimmed_mg_pred_df = mg_pred_df[['iqr_anomaly']]
        trimmed_mg_pred_df.rename(columns={'iqr_anomaly': 'gmm_pred'}, inplace=True)
    return gmm_filter_df  # , trimmed_mg_pred_df


def runKMEANS(recruit_contigs_df, sag_id, std_merge_df):
    temp_cat_df = std_merge_df.copy()
    last_len = 0
    while temp_cat_df.shape[0] != last_len:
        last_len = temp_cat_df.shape[0]
        clusters = 10 if last_len >= 10 else last_len
        kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=42).fit(temp_cat_df.values)
        clust_labels = kmeans.labels_
        clust_df = pd.DataFrame(zip(temp_cat_df.index.values, clust_labels),
                                columns=['subcontig_id', 'kmeans_clust']
                                )
        recruit_clust_df = clust_df.loc[clust_df['subcontig_id'].isin(list(recruit_contigs_df.index))]
        subset_clust_df = clust_df.loc[clust_df['kmeans_clust'].isin(
            list(recruit_clust_df['kmeans_clust'].unique())
        )]
        subset_clust_df['kmeans_pred'] = 1
        temp_cat_df = temp_cat_df.loc[temp_cat_df.index.isin(list(subset_clust_df['subcontig_id']))]
        if temp_cat_df.shape[0] == 0:
            break
    if temp_cat_df.shape[0] == 0:
        kmeans_pass_list = []
    else:
        cat_clust_df = subset_clust_df.copy()  # pd.concat(block_list)
        std_id_df = pd.DataFrame(std_merge_df.index.values, columns=['subcontig_id'])
        std_id_df['contig_id'] = [x.rsplit('_', 1)[0] for x in std_id_df['subcontig_id']]
        cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
        sub_std_df = std_id_df.loc[std_id_df['contig_id'].isin(list(cat_clust_df['contig_id']))]
        std_clust_df = sub_std_df.merge(cat_clust_df, on=['subcontig_id', 'contig_id'], how='outer')
        std_clust_df.fillna(-1, inplace=True)
        pred_df = std_clust_df[['subcontig_id', 'contig_id', 'kmeans_pred']]
        val_perc = pred_df.groupby('contig_id')['kmeans_pred'].value_counts(normalize=True).reset_index(name='percent')
        pos_perc = val_perc.loc[val_perc['kmeans_pred'] == 1]
        major_df = pos_perc.loc[pos_perc['percent'] >= 0.95]
        major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
        kmeans_pass_list = []
        for md_nm in major_pred_df['subcontig_id']:
            kmeans_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return kmeans_pass_list


def iqr_bounds(scores, k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound = (q1 - k * iqr)
    upper_bound = (q3 + k * iqr)
    return lower_bound, upper_bound


def subset_tetras(p):
    mg_tetra_dict, mh_sag_df, abund_sag_df, sag_id = p
    if mh_sag_df.shape[0] != 0:
        sag_df_list = [mg_tetra_dict[x] for x in set(mh_sag_df['contig_id'])]
        sag_tetra_df = pd.concat(sag_df_list)
        del sag_tetra_df['contig_id']
        mg_tetra_df_list = [mg_tetra_dict[x] for x in set(abund_sag_df['contig_id'])]
        mg_tetra_filter_df = pd.concat(mg_tetra_df_list)
        del mg_tetra_filter_df['contig_id']
    else:
        sag_tetra_df = None
        mg_tetra_filter_df = None
    return sag_id, sag_tetra_df, mg_tetra_filter_df


def calc_components(sag_tetra_df):
    # logging.info('Calculating AIC/BIC for GMM components\n')
    sag_train_vals = [1 for x in sag_tetra_df.index]
    n_components = range(1, 100, 1)
    models = [GMM(n, random_state=42) for n in n_components]
    bics = []
    aics = []
    min_bic = np.inf
    bic = np.inf
    bic_counter = 0
    for i, model in enumerate(models):
        n_comp = n_components[i]
        if ((bic_counter <= 5) & (sag_tetra_df.shape[0] >= n_comp) & (sag_tetra_df.shape[0] >= 2)):
            bic = model.fit(sag_tetra_df.values,
                            sag_train_vals).bic(sag_tetra_df.values
                                                )
            bics.append(bic)
            if min_bic > bic:
                min_bic = bic
                bic_counter = 0
            else:
                bic_counter += 1
    if min_bic != np.inf:
        min_bic_comp = n_components[bics.index(min_bic)]
    else:
        min_bic_comp = None
    return min_bic_comp


def filter_tetras(sag_id, mg_headers, tetra_id, tetra_df):
    # tetra_df = tetra_df_dict[tetra_id]
    # Count # of subcontigs recruited to each SAG
    cnt_df = tetra_df.groupby(['sag_id', 'contig_id']).count().reset_index()
    cnt_df.columns = ['sag_id', 'contig_id', 'subcontig_recruits']
    # Build subcontig count for each MG contig
    mg_contig_list = [x.rsplit('_', 1)[0] for x in mg_headers]
    mg_tot_df = pd.DataFrame(zip(mg_contig_list, mg_headers),
                             columns=['contig_id', 'subcontig_id'])
    mg_tot_cnt_df = mg_tot_df.groupby(['contig_id']).count().reset_index()
    mg_tot_cnt_df.columns = ['contig_id', 'subcontig_total']
    mg_recruit_df = cnt_df.merge(mg_tot_cnt_df, how='left', on='contig_id')
    mg_recruit_df[tetra_id + '_p'] = mg_recruit_df['subcontig_recruits'] / \
                                     mg_recruit_df['subcontig_total']
    mg_recruit_df.sort_values(by=tetra_id + '_p', ascending=False, inplace=True)
    # Only pass contigs that have the magjority of subcontigs recruited (>= N%)
    thresh_dict = {'gmm': 0.50, 'svm': 0.00, 'iso': 0.74}
    tetra_w = 1 - thresh_dict[tetra_id]
    tot_W = sum([1 - thresh_dict[x] for x in thresh_dict.keys()])
    mg_recruit_filter_df = mg_recruit_df.loc[
        mg_recruit_df[tetra_id + '_p'] > thresh_dict[tetra_id]
        ]
    mg_recruit_filter_df[tetra_id + '_s'] = ((mg_recruit_filter_df[tetra_id + '_p'] - \
                                              thresh_dict[tetra_id]) / tetra_w
                                             )
    mg_recruit_filter_df[tetra_id + '_w'] = (mg_recruit_filter_df[tetra_id + '_s'] * \
                                             (tetra_w / tot_W)
                                             )
    tetra_max_df = mg_tot_df.merge(mg_recruit_filter_df, on=['contig_id'], how='inner')
    # tetra_max_df = mg_tot_df[mg_tot_df['contig_id'].isin(
    #    list(mg_recruit_filter_df['contig_id'])
    # )]
    tetra_max_df['sag_id'] = sag_id
    tetra_max_df = tetra_max_df[['sag_id', 'subcontig_id', 'contig_id', tetra_id + '_p',
                                 tetra_id + '_s', tetra_id + '_w'
                                 ]]
    return tetra_max_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='uses tetrenucleotide Hz to recruit metaG reads to SAGs')
    parser.add_argument(
        '--tetra_path', help='path to tetrenucleotide output directory',
        required=True
    )
    parser.add_argument(
        '--sag_sub_file',
        help='path to SAG subcontigs file', required=True
    )
    parser.add_argument(
        '--mg_sub_file',
        help='path to metagenome subcontigs file', required=True
    )
    parser.add_argument(
        '--abund_df',
        help='path to output dataframe from abundance recruiter', required=True
    )
    parser.add_argument(
        '--per_pass',
        help='pass percentage of subcontigs to pass complete contig', required=True,
        default='0.01'
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Prints a more verbose runtime log"
                        )
    args = parser.parse_args()
    # set args
    tra_path = args.tetra_path
    sag_sub_file = args.sag_sub_file
    mg_sub_file = args.mg_sub_file
    abund_recruit_file = args.abund_df
    per_pass = float(args.per_pass)

    s_log.prep_logging("tetra_log.txt", args.verbose)
    sag_id = basename(sag_sub_file).rsplit('.', 2)[0]
    mg_id = basename(mg_sub_file).rsplit('.', 2)[0]
    abund_recruit_df = pd.read_csv(abund_recruit_file, header=0, sep='\t')
    logging.info('Starting Tetranucleotide Recruitment Step\n')
    run_tetra_recruiter(tra_path, [[sag_id, sag_sub_file]], [mg_id, mg_sub_file],
                        abund_recruit_df, per_pass)


