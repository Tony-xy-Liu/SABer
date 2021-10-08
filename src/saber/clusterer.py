#!/usr/bin/env python

import multiprocessing
import warnings
from os.path import join as o_join
from pathlib import Path

import hdbscan
import pandas as pd
import umap
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def runOCSVM(tc_df, mg_df, tc_id, gamma, nu):
    # fit OCSVM
    clf = svm.OneClassSVM()  # nu=nu, gamma=gamma)
    clf.fit(tc_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['nu'] = 'default'  # nu
    pred_df['gamma'] = 'default'  # gamma
    pred_df['sag_id'] = tc_id
    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

    return pred_df


def runISOF(sag_df, mg_df, sag_id, contam=0, estim=10, max_samp='auto'):
    # fit IsoForest
    clf = IsolationForest(random_state=42, contamination=contam, n_estimators=estim,
                          max_samples=max_samp
                          )
    clf.fit(sag_df.values)
    sag_pred = clf.predict(sag_df.values)
    sag_score = clf.decision_function(sag_df.values)
    sag_pred_df = pd.DataFrame(data=sag_pred, index=sag_df.index.values,
                               columns=['anomaly'])
    sag_pred_df.loc[sag_pred_df['anomaly'] == 1, 'anomaly'] = 0
    sag_pred_df.loc[sag_pred_df['anomaly'] == -1, 'anomaly'] = 1
    sag_pred_df['scores'] = sag_score
    lower_bound, upper_bound = iqr_bounds(sag_pred_df['scores'], k=0.5)

    mg_pred = clf.predict(mg_df.values)
    mg_score = clf.decision_function(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'anomaly']
                           )
    pred_df.loc[pred_df['anomaly'] == 1, 'anomaly'] = 0
    pred_df.loc[pred_df['anomaly'] == -1, 'anomaly'] = 1
    pred_df['scores'] = mg_score
    pred_df['iqr_anomaly'] = 0
    pred_df['iqr_anomaly'] = (pred_df['scores'] < lower_bound) | \
                             (pred_df['scores'] > upper_bound)
    pred_df['iqr_anomaly'] = pred_df['iqr_anomaly'].astype(int)
    pred_df['sag_id'] = sag_id
    pred_df['pred'] = pred_df['iqr_anomaly'] == 0
    pred_df['pred'] = pred_df['pred'].astype(int)

    return pred_df


def iqr_bounds(scores, k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound = (q1 - k * iqr)
    upper_bound = (q3 + k * iqr)
    return lower_bound, upper_bound


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
    cat_clust_df = subset_clust_df.copy()  # pd.concat(block_list)
    std_id_df = pd.DataFrame(std_merge_df.index.values, columns=['subcontig_id'])
    std_id_df['contig_id'] = [x.rsplit('_', 1)[0] for x in std_id_df['subcontig_id']]
    cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
    sub_std_df = std_id_df.loc[std_id_df['contig_id'].isin(list(cat_clust_df['contig_id']))]
    std_clust_df = sub_std_df.merge(cat_clust_df, on=['subcontig_id', 'contig_id'], how='outer')
    std_clust_df.fillna(-1, inplace=True)
    pred_df = std_clust_df[['subcontig_id', 'contig_id', 'kmeans_pred']]
    val_perc = pred_df.groupby('contig_id')['kmeans_pred'
    ].value_counts(normalize=True).reset_index(name='percent')
    pos_perc = val_perc.loc[val_perc['kmeans_pred'] == 1]
    major_df = pos_perc.loc[pos_perc['percent'] >= 0.51]
    major_pred_df = pred_df.loc[pred_df['contig_id'].isin(major_df['contig_id'])]
    std_clust_pred_df = std_clust_df.merge(major_pred_df, on=['subcontig_id', 'contig_id'],
                                           how='left'
                                           )
    filter_clust_pred_df = std_clust_pred_df.loc[std_clust_pred_df['kmeans_pred_x'] == 1]
    kmeans_pass_list = []
    for md_nm in major_pred_df['subcontig_id']:
        kmeans_pass_list.append([sag_id, md_nm, md_nm.rsplit('_', 1)[0]])
    return kmeans_pass_list, filter_clust_pred_df


def recruitOCSVM(p):
    merge_df, mh_trusted_df, no_noise_df, noise_df, sag_id = p
    # Subset trusted contigs by SAG
    sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0')
    trusted_contigs_list = sub_trusted_df['contig_id'].unique()
    # Gather trusted contigs subset from cluster and noise DFs
    sub_denovo_df = no_noise_df.query('contig_id in @trusted_contigs_list and '
                                      'probabilities >= 1.0'
                                      )
    sub_noise_df = noise_df.query('contig_id in @trusted_contigs_list')
    # Gather all contigs associated with trusted clusters
    trust_clust_list = list(sub_denovo_df['best_label'].unique())
    sub_label_df = no_noise_df.query('best_label in @trust_clust_list')
    sub_contig_list = list(set(list(sub_label_df['contig_id']) +
                               list(sub_noise_df['contig_id']))
                           )
    tc_contig_list = list(sub_trusted_df['contig_id'].unique())
    merge_df['contig_id'] = [x.rsplit('_', 1)[0] for x in merge_df.index.values]
    tc_feat_df = merge_df.query('contig_id in @tc_contig_list')
    mg_feat_df = merge_df.copy()  # .query('contig_id in @sub_contig_list')
    mg_feat_df.drop(columns=['contig_id'], inplace=True)
    tc_feat_df.drop(columns=['contig_id'], inplace=True)
    major_df = False
    if (tc_feat_df.shape[0] != 0) & (mg_feat_df.shape[0] != 0):
        # Run KMEANS first
        kmeans_pass_list, kclusters_df = runKMEANS(tc_feat_df, sag_id, mg_feat_df)
        kmeans_pass_df = pd.DataFrame(kmeans_pass_list,
                                      columns=['sag_id', 'subcontig_id', 'contig_id']
                                      )
        nonrecruit_kmeans_df = mg_feat_df.loc[kmeans_pass_df['subcontig_id']]
        ocsvm_recruit_df = runOCSVM(tc_feat_df, nonrecruit_kmeans_df, sag_id, None, None)
        val_perc = ocsvm_recruit_df.groupby('contig_id')['pred'].value_counts(
            normalize=True).reset_index(name='percent')
        pos_perc = val_perc.loc[val_perc['pred'] == 1]
        major_df = pos_perc.copy()  # .loc[pos_perc['percent'] >= 0.51]
        major_df['sag_id'] = sag_id

    return major_df

def runClusterer(mg_id, clst_path, cov_file, tetra_file, minhash_dict,
                 nthreads
                 ):  # TODO: need to add multithreading where ever possible
    # Convert CovM to UMAP feature table
    set_init = 'spectral'
    cov_emb = Path(o_join(clst_path, mg_id + '.denovo.covm_emb.tsv'))
    if not cov_emb.is_file():
        print('Building UMAP embedding for Coverage...')
        cov_df = pd.read_csv(cov_file, header=0, sep='\t', index_col='contigName')
        n_neighbors = 10
        # COV sometimes crashes when init='spectral', so fall back on 'random' when that happens.
        try:
            clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0,
                                              n_components=len(cov_df.columns),
                                              random_state=42, metric='manhattan', init=set_init
                                              ).fit_transform(cov_df)
        except:
            print('Resetting UMAP initialization to random to avoid warning...')
            tmp_init = 'random'
            clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0,
                                              n_components=len(cov_df.columns),
                                              random_state=42, metric='manhattan', init=tmp_init
                                              ).fit_transform(
                cov_df)  # TODO: Spectral works better for De novo, Random for Anchored.
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=cov_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(cov_emb, sep='\t', index=False)
    # Convert Tetra to UMAP feature table
    tetra_emb = Path(o_join(clst_path, mg_id + '.denovo.tetra_emb.tsv'))
    if not tetra_emb.is_file():
        print('Building UMAP embedding for Tetra Hz...')
        tetra_df = pd.read_csv(tetra_file, header=0, sep='\t', index_col='contig_id')
        n_neighbors = 10
        clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=40,
                                          random_state=42, metric='manhattan', init=set_init
                                          ).fit_transform(tetra_df)
        umap_feat_df = pd.DataFrame(clusterable_embedding, index=tetra_df.index.values)
        umap_feat_df.reset_index(inplace=True)
        umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
        umap_feat_df.to_csv(tetra_emb, sep='\t', index=False)
    # Merge Coverage and Tetra Embeddings
    merged_emb = Path(o_join(clst_path, mg_id + '.denovo.merged_emb.tsv'))
    if not merged_emb.is_file():
        print('Merging Tetra and Coverage Embeddings...')
        tetra_feat_df = pd.read_csv(tetra_emb, sep='\t', header=0, index_col='subcontig_id')
        tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df.index.values]
        tetra_feat_df.columns = [str(x) + '_tetra' for x in tetra_feat_df.columns]
        # load covm file
        cov_feat_df = pd.read_csv(cov_emb, sep='\t', header=0)
        cov_feat_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        cov_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_feat_df['subcontig_id']]
        cov_feat_df.set_index('subcontig_id', inplace=True)
        cov_feat_df.columns = [str(x) + '_cov' for x in cov_feat_df.columns]
        merge_df = tetra_feat_df.join(cov_feat_df)
        merge_df.drop(columns=['contig_id_tetra', 'contig_id_cov'], inplace=True)
        tetra_feat_df.drop(columns=['contig_id_tetra'], inplace=True)
        cov_feat_df.drop(columns=['contig_id_cov'], inplace=True)
        merge_df.to_csv(merged_emb, sep='\t')
    else:
        print('Loading Merged Embedding...')
        merge_df = pd.read_csv(merged_emb, sep='\t', header=0, index_col='subcontig_id')
    denovo_out_file = Path(o_join(clst_path, mg_id + '.denovo_clusters.tsv'))
    noise_out_file = Path(o_join(clst_path, mg_id + '.noise.tsv'))
    if not denovo_out_file.is_file():
        print('Performing De Novo Clustering...')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, cluster_selection_method='eom',
                                    prediction_data=True, cluster_selection_epsilon=0,
                                    min_samples=25
                                    ).fit(merge_df.values)

        cluster_labels = clusterer.labels_
        cluster_probs = clusterer.probabilities_
        cluster_outlier = clusterer.outlier_scores_

        cluster_df = pd.DataFrame(zip(merge_df.index.values, cluster_labels, cluster_probs,
                                      cluster_outlier),
                                  columns=['subcontig_id', 'label', 'probabilities',
                                           'outlier_score']
                                  )
        cluster_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cluster_df['subcontig_id']]

        cluster_df.to_csv(o_join(clst_path, mg_id + '.hdbscan.tsv'),
                          sep='\t', index=False
                          )

        print('Denoising Clusters...')
        ns_ratio_list = []
        for contig in tqdm(list(cluster_df['contig_id'].unique())):
            sub_df = cluster_df.query('contig_id == @contig')
            noise_cnt = sub_df.query('label == -1').shape[0]
            signal_cnt = sub_df.query('label != -1').shape[0]
            ns_ratio = (noise_cnt / (noise_cnt + signal_cnt)) * 100
            prob_df = sub_df.groupby(['label'])[['probabilities']].max().reset_index()
            best_ind = prob_df['probabilities'].argmax()
            best_label = prob_df['label'].iloc[best_ind]
            best_prob = prob_df['probabilities'].iloc[best_ind]
            ns_ratio_list.append([contig, best_label])

        ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'best_label'])
        cluster_ns_df = cluster_df.merge(ns_ratio_df, on='contig_id', how='left')
        no_noise_df = cluster_ns_df.query('best_label != -1')  # 'best_prob >= 0.51')
        noise_df = cluster_ns_df.query('best_label == -1')  # 'best_prob < 0.51')
        if no_noise_df.empty:
            #  TODO: fix this or print a warning message to user :)
            no_noise_df = noise_df.copy()
        no_noise_df.to_csv(denovo_out_file, sep='\t', index=False)
        noise_df.to_csv(noise_out_file, sep='\t', index=False)
    else:
        print('Loading De Novo Clusters...')
        no_noise_df = pd.read_csv(denovo_out_file, header=0, sep='\t')
        noise_df = pd.read_csv(noise_out_file, header=0, sep='\t')

    trust_out_file = Path(o_join(clst_path, mg_id + '.trusted_clusters.tsv'))
    if minhash_dict and not trust_out_file.is_file():
        print('Anchored Binning Starting with Trusted Contigs...')
        # Convert CovM to UMAP feature table
        set_init = 'random'
        cov_emb = Path(o_join(clst_path, mg_id + '.anchored.covm_emb.tsv'))
        if not cov_emb.is_file():
            print('Building UMAP embedding for Coverage...')
            cov_df = pd.read_csv(cov_file, header=0, sep='\t', index_col='contigName')
            n_neighbors = 10
            clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0,
                                              n_components=len(cov_df.columns),
                                              random_state=42, metric='manhattan', init=set_init
                                              ).fit_transform(cov_df)
            umap_feat_df = pd.DataFrame(clusterable_embedding, index=cov_df.index.values)
            umap_feat_df.reset_index(inplace=True)
            umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
            umap_feat_df.to_csv(cov_emb, sep='\t', index=False)
        # Convert Tetra to UMAP feature table
        tetra_emb = Path(o_join(clst_path, mg_id + '.anchored.tetra_emb.tsv'))
        if not tetra_emb.is_file():
            print('Building UMAP embedding for Tetra Hz...')
            tetra_df = pd.read_csv(tetra_file, header=0, sep='\t', index_col='contig_id')
            n_neighbors = 10
            clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=40,
                                              random_state=42, metric='manhattan', init=set_init
                                              ).fit_transform(tetra_df)
            umap_feat_df = pd.DataFrame(clusterable_embedding, index=tetra_df.index.values)
            umap_feat_df.reset_index(inplace=True)
            umap_feat_df.rename(columns={'index': 'subcontig_id'}, inplace=True)
            umap_feat_df.to_csv(tetra_emb, sep='\t', index=False)
        # Merge Coverage and Tetra Embeddings
        anchor_emb = Path(o_join(clst_path, mg_id + '.anchored.merged_emb.tsv'))
        if not anchor_emb.is_file():
            print('Merging Tetra and Coverage Embeddings...')
            tetra_feat_df = pd.read_csv(tetra_emb, sep='\t', header=0, index_col='subcontig_id')
            tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df.index.values]
            # load covm file
            cov_feat_df = pd.read_csv(cov_emb, sep='\t', header=0)
            cov_feat_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
            cov_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_feat_df['subcontig_id']]
            cov_feat_df.set_index('subcontig_id', inplace=True)
            anchor_df = tetra_feat_df.join(cov_feat_df, lsuffix='_tetra', rsuffix='_cov')
            anchor_df.drop(columns=['contig_id_tetra', 'contig_id_cov'], inplace=True)
            tetra_feat_df.drop(columns=['contig_id'], inplace=True)
            cov_feat_df.drop(columns=['contig_id'], inplace=True)
            anchor_df.to_csv(anchor_emb, sep='\t')
        else:
            print('Loading Merged Embedding...')
            anchor_df = pd.read_csv(anchor_emb, sep='\t', header=0, index_col='subcontig_id')

        # Group clustered and noise contigs by trusted contigs
        print('Re-grouping with Trusted Contigs...')
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        label_max_list = []
        contig_max_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            # Subset trusted contigs by SAG
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0')
            trusted_contigs_list = sub_trusted_df['contig_id'].unique()
            # Gather trusted contigs subset from cluster and noise DFs
            sub_denovo_df = no_noise_df.query('contig_id in @trusted_contigs_list and '
                                              'probabilities >= 1.0'
                                              )
            sub_noise_df = noise_df.query('contig_id in @trusted_contigs_list')
            # Gather all contigs associated with trusted clusters
            trust_clust_list = list(sub_denovo_df['best_label'].unique())
            sub_label_df = no_noise_df.query('best_label in @trust_clust_list')
            # Get average max jaccard for each cluster
            label_mh_df = sub_label_df.merge(sub_trusted_df, on='contig_id', how='left')
            label_max_df = label_mh_df.groupby(['best_label'])[['jacc_sim']].mean().reset_index()
            label_max_df['sag_id'] = sag_id
            label_max_list.append(label_max_df)
            # Get max jaccard for noise labeled contigs
            if sub_noise_df.shape[0] != 0:
                noise_mh_df = sub_noise_df.merge(sub_trusted_df, on='contig_id', how='left')
                noise_max_df = noise_mh_df.groupby(['contig_id'])[['jacc_sim']].mean().reset_index()
                noise_max_df['sag_id'] = sag_id
                contig_max_list.append(noise_max_df)

        sag_label_df = pd.concat(label_max_list)
        sag_contig_df = pd.concat(contig_max_list)
        sag_label_best_df = sag_label_df.sort_values(by='jacc_sim', ascending=False
                                                     ).drop_duplicates(subset='best_label')
        sag_contig_best_df = sag_contig_df.sort_values(by='jacc_sim', ascending=False
                                                       ).drop_duplicates(subset='contig_id')
        sag_label_list = []
        for index, row in sag_label_best_df.iterrows():
            best_label = row['best_label']
            jacc_sim = row['jacc_sim']
            sub_sag_label_df = sag_label_df.query('best_label == @best_label and '
                                                  'jacc_sim == @jacc_sim'
                                                  )
            sag_label_list.append(sub_sag_label_df)

        sag_contig_list = []
        for index, row in sag_contig_best_df.iterrows():
            sag_id = row['sag_id']
            jacc_sim = row['jacc_sim']
            sub_sag_contig_df = sag_contig_df.query('sag_id == @sag_id and '
                                                    'jacc_sim >= @jacc_sim'
                                                    )
            sag_contig_list.append(sub_sag_contig_df)

        if sag_label_list:
            label_pruned_df = pd.concat(sag_label_list)
            sag_denovo_df = label_pruned_df.merge(no_noise_df, on='best_label', how='left')
        else:
            sag_denovo_df = pd.DataFrame(columns=['sag_id', 'contig_id'])

        contig_pruned_df = pd.concat(sag_contig_list)
        sag_noise_df = contig_pruned_df.merge(noise_df, on='contig_id', how='left')

        print('Building Trusted Clusters...')
        trust_recruit_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            trust_cols = ['sag_id', 'contig_id']
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0'
                                                 )[trust_cols]
            sub_denovo_df = sag_denovo_df.query('sag_id == @sag_id')[trust_cols]
            sub_noise_df = sag_noise_df.query('sag_id == @sag_id')[trust_cols]
            trust_cat_df = pd.concat([sub_trusted_df, sub_denovo_df, sub_noise_df]
                                     ).drop_duplicates()
            trust_recruit_list.append(trust_cat_df)

        trust_recruit_df = pd.concat(trust_recruit_list)
        trust_recruit_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
        trust_recruit_df.to_csv(trust_out_file, sep='\t', index=False)
    elif trust_out_file.is_file():
        print('Trust Clusters already exist...')
        trust_recruit_df = pd.read_csv(trust_out_file, sep='\t', header=0)
    else:
        print('No Trusted Contigs Provided...')
        trust_recruit_df = False

    # Run OC-SVM recruiting
    ocsvm_out_file = Path(o_join(clst_path, mg_id + '.ocsvm_clusters.tsv'))
    anchor_emb = Path(o_join(clst_path, mg_id + '.anchored.merged_emb.tsv'))
    if minhash_dict and not ocsvm_out_file.is_file():
        print('Performing Anchored Recruitment with OC-SVM...')
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        anchor_df = pd.read_csv(anchor_emb, sep='\t', header=0, index_col='subcontig_id')
        pool = multiprocessing.Pool(processes=nthreads)
        arg_list = []
        for sag_id in mh_trusted_df['sag_id'].unique():
            arg_list.append([anchor_df, mh_trusted_df, no_noise_df, noise_df, sag_id])
        ocsvm_recruit_list = []
        results = pool.imap_unordered(recruitOCSVM, arg_list)
        for i, output in tqdm(enumerate(results, 1)):
            if isinstance(output, pd.DataFrame):
                ocsvm_recruit_list.append(output)
        pool.close()
        pool.join()
        ocsvm_contig_df = pd.concat(ocsvm_recruit_list)
        ocsvm_contig_best_df = ocsvm_contig_df.sort_values(by='percent', ascending=False
                                                           ).drop_duplicates(subset='contig_id')
        sag_contig_list = []
        for index, row in tqdm(ocsvm_contig_best_df.iterrows()):
            sag_id = row['sag_id']
            percent = row['percent']
            sub_sag_contig_df = ocsvm_contig_df.query('sag_id == @sag_id and '
                                                      'percent >= @percent'
                                                      )
            sag_contig_list.append(sub_sag_contig_df)
        contig_pruned_df = pd.concat(sag_contig_list)

        print('Building OC-SVM Clusters...')
        ocsvm_clust_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            trust_cols = ['sag_id', 'contig_id']
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0'
                                                 )[trust_cols]
            sub_ocsvm_df = contig_pruned_df.query('sag_id == @sag_id')[trust_cols]
            ocsvm_cat_df = pd.concat([sub_trusted_df, sub_ocsvm_df]).drop_duplicates()
            ocsvm_clust_list.append(ocsvm_cat_df)

        ocsvm_clust_df = pd.concat(ocsvm_clust_list)
        ocsvm_clust_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
        ocsvm_clust_df.to_csv(ocsvm_out_file, sep='\t', index=False)
    elif ocsvm_out_file.is_file():
        print('OC-SVM Clusters already exist...')
        ocsvm_clust_df = pd.read_csv(ocsvm_out_file, sep='\t', header=0)
    else:
        ocsvm_clust_df = False

    # Find intersection of HDBSCAN and OC-SVM
    inter_out_file = Path(o_join(clst_path, mg_id + '.inter_clusters.tsv'))
    if minhash_dict and not inter_out_file.is_file():
        print('Combining Recruits from HDBSCAN and OC-SVM...')
        inter_clust_list = []
        for best_label in tqdm(trust_recruit_df['best_label'].unique()):
            sub_scan_df = trust_recruit_df.query('best_label == @best_label')
            sub_svm_df = ocsvm_clust_df.query('best_label == @best_label')
            intersect = set(sub_scan_df['contig_id']).intersection(set(sub_svm_df['contig_id']))
            best_inter_list = [(best_label, x) for x in intersect]
            best_inter_df = pd.DataFrame(best_inter_list, columns=['best_label', 'contig_id'])
            inter_clust_list.append(best_inter_df)
        inter_clust_df = pd.concat(inter_clust_list)
        inter_clust_df.to_csv(inter_out_file, sep='\t', index=False)
    elif inter_out_file.is_file():
        print('Combined Clusters already exist...')
        inter_clust_df = pd.read_csv(inter_out_file, sep='\t', header=0)
    else:
        inter_clust_df = False

    #############################################################################################

    # Convert CovM to UMAP feature table
    cov_emb = Path(o_join(clst_path, mg_id + '.denovo.covm_emb.tsv'))
    deit_out_file = Path(o_join(clst_path, mg_id + '.deit_clusters.tsv'))
    noit_out_file = Path(o_join(clst_path, mg_id + '.noit.tsv'))
    if not deit_out_file.is_file():
        print('Performing De Novo Clustering (iteration 2)...')
        cov_feat_df = pd.read_csv(cov_emb, sep='\t', header=0)
        cov_feat_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
        cov_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_feat_df['subcontig_id']]
        cov_feat_df.set_index('subcontig_id', inplace=True)
        cov_feat_df.columns = [str(x) + '_cov' for x in cov_feat_df.columns]
        # cov_feat_df.drop(columns=['contig_id_cov'], inplace=True)
        clust_list = []
        for clust in tqdm(no_noise_df['best_label'].unique()):
            sub_denovo_df = no_noise_df.query('best_label == @clust')
            deno_contig_list = list(sub_denovo_df['contig_id'])
            sub_cov_df = cov_feat_df.query('contig_id_cov in @deno_contig_list')
            sub_cov_df.drop(columns=['contig_id_cov'], inplace=True)
            k = 25
            if sub_cov_df.shape[0] < k:
                k = sub_cov_df.shape[0]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=100, cluster_selection_method='eom',
                                        prediction_data=True, cluster_selection_epsilon=0,
                                        min_samples=k, allow_single_cluster=True
                                        ).fit(sub_cov_df.values)
            cluster_labels = clusterer.labels_
            cluster_probs = clusterer.probabilities_
            cluster_outlier = clusterer.outlier_scores_
            joined_labels = [clust for x in cluster_labels]
            cluster_df = pd.DataFrame(zip(sub_cov_df.index.values, cluster_labels, cluster_probs,
                                          cluster_outlier, joined_labels),
                                      columns=['subcontig_id', 'sub_label', 'probabilities',
                                               'outlier_score', 'label']
                                      )
            clust_list.append(cluster_df)
        cat_clust_df = pd.concat(clust_list)
        cat_clust_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cat_clust_df['subcontig_id']]
        cat_clust_df.to_csv(o_join(clst_path, mg_id + '.hdbit.tsv'),
                            sep='\t', index=False
                            )
        print('Denoising Clusters (iteration 2)...')
        ns_ratio_list = []
        for contig in tqdm(list(cat_clust_df['contig_id'].unique())):
            sub_df = cat_clust_df.query('contig_id == @contig')
            noise_cnt = sub_df.query('sub_label == -1').shape[0]
            signal_cnt = sub_df.query('sub_label != -1').shape[0]
            ns_ratio = (noise_cnt / (noise_cnt + signal_cnt)) * 100
            prob_df = sub_df.groupby(['label', 'sub_label'])[['probabilities']].max().reset_index()
            best_ind = prob_df['probabilities'].argmax()
            best_sub_label = prob_df['sub_label'].iloc[best_ind]
            clust_label = prob_df['label'].iloc[best_ind]
            best_label = str(clust_label) + '|' + str(best_sub_label)
            best_prob = prob_df['probabilities'].iloc[best_ind]
            ns_ratio_list.append([contig, clust_label, best_sub_label, best_label])
        ns_ratio_df = pd.DataFrame(ns_ratio_list, columns=['contig_id', 'clust_label',
                                                           'best_sub_label', 'best_label']
                                   )
        cluster_ns_df = cat_clust_df.merge(ns_ratio_df, on='contig_id', how='left')
        no_noit_df = cluster_ns_df.query('best_sub_label != -1')
        noit_df = cluster_ns_df.query('best_sub_label == -1')
        if no_noit_df.empty:
            #  TODO: fix this or print a warning message to user :)
            no_noit_df = noit_df.copy()
        no_noit_df.to_csv(deit_out_file, sep='\t', index=False)
        noit_df.to_csv(noit_out_file, sep='\t', index=False)
    else:
        print('Loading De Novo Clusters (iteration 2)...')
        no_noit_df = pd.read_csv(deit_out_file, header=0, sep='\t')
        noit_df = pd.read_csv(noit_out_file, header=0, sep='\t')

    #############################################################################################

    trit_out_file = Path(o_join(clst_path, mg_id + '.trusted_it.tsv'))
    anchor_emb = Path(o_join(clst_path, mg_id + '.anchored.merged_emb.tsv'))
    if minhash_dict and not trit_out_file.is_file():
        # Group clustered and noise contigs by trusted contigs
        print('Re-grouping with Trusted Contigs...')
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        label_max_list = []
        contig_max_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            # Subset trusted contigs by SAG
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0')
            trusted_contigs_list = sub_trusted_df['contig_id'].unique()
            # Gather trusted contigs subset from cluster and noise DFs
            sub_denovo_df = no_noit_df.query('contig_id in @trusted_contigs_list and '
                                             'probabilities >= 1.0'
                                             )
            sub_noise_df = noit_df.query('contig_id in @trusted_contigs_list')
            # Gather all contigs associated with trusted clusters
            trust_clust_list = list(sub_denovo_df['best_label'].unique())
            sub_label_df = no_noit_df.query('best_label in @trust_clust_list')
            # Get average max jaccard for each cluster
            label_mh_df = sub_label_df.merge(sub_trusted_df, on='contig_id', how='left')
            label_max_df = label_mh_df.groupby(['best_label'])[['jacc_sim']].mean().reset_index()
            label_max_df['sag_id'] = sag_id
            label_max_list.append(label_max_df)
            # Get max jaccard for noise labeled contigs
            if sub_noise_df.shape[0] != 0:
                noise_mh_df = sub_noise_df.merge(sub_trusted_df, on='contig_id', how='left')
                noise_max_df = noise_mh_df.groupby(['contig_id'])[['jacc_sim']].mean().reset_index()
                noise_max_df['sag_id'] = sag_id
                contig_max_list.append(noise_max_df)

        sag_label_df = pd.concat(label_max_list)
        if contig_max_list:
            sag_contig_df = pd.concat(contig_max_list)
        else:
            sag_contig_df = pd.DataFrame(columns=['sag_id', 'jacc_sim', 'contig_id',
                                                  'best_label']
                                         )
        sag_label_best_df = sag_label_df.sort_values(by='jacc_sim', ascending=False
                                                     ).drop_duplicates(subset='best_label')
        sag_contig_best_df = sag_contig_df.sort_values(by='jacc_sim', ascending=False
                                                       ).drop_duplicates(subset='contig_id')
        sag_label_list = []
        for index, row in sag_label_best_df.iterrows():
            best_label = row['best_label']
            jacc_sim = row['jacc_sim']
            sub_sag_label_df = sag_label_df.query('best_label == @best_label and '
                                                  'jacc_sim == @jacc_sim'
                                                  )
            sag_label_list.append(sub_sag_label_df)

        sag_contig_list = []
        for index, row in sag_contig_best_df.iterrows():
            sag_id = row['sag_id']
            jacc_sim = row['jacc_sim']
            sub_sag_contig_df = sag_contig_df.query('sag_id == @sag_id and '
                                                    'jacc_sim >= @jacc_sim'
                                                    )
            sag_contig_list.append(sub_sag_contig_df)

        if sag_label_list:
            label_pruned_df = pd.concat(sag_label_list)
            sag_denovo_df = label_pruned_df.merge(no_noit_df, on='best_label', how='left')
        else:
            sag_denovo_df = pd.DataFrame(columns=['sag_id', 'contig_id'])
        if sag_contig_list:
            contig_pruned_df = pd.concat(sag_contig_list)
            sag_noise_df = contig_pruned_df.merge(noit_df, on='contig_id', how='left')
        else:
            sag_noise_df = pd.DataFrame(columns=['sag_id', 'contig_id'])

        print('Building Trusted Clusters...')
        trust_recruit_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            trust_cols = ['sag_id', 'contig_id']
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0'
                                                 )[trust_cols]
            sub_denovo_df = sag_denovo_df.query('sag_id == @sag_id')[trust_cols]
            sub_noise_df = sag_noise_df.query('sag_id == @sag_id')[trust_cols]
            trust_cat_df = pd.concat([sub_trusted_df, sub_denovo_df, sub_noise_df]
                                     ).drop_duplicates()
            trust_recruit_list.append(trust_cat_df)

        trust_recruit_df = pd.concat(trust_recruit_list)
        trust_recruit_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
        trust_recruit_df.to_csv(trit_out_file, sep='\t', index=False)
    elif trit_out_file.is_file():
        print('Trust Clusters already exist...')
        trust_recruit_df = pd.read_csv(trit_out_file, sep='\t', header=0)
    else:
        print('No Trusted Contigs Provided...')
        trust_recruit_df = False

    #############################################################################################

    # Run OC-SVM recruiting
    ocsvm_it_file = Path(o_join(clst_path, mg_id + '.ocsvm_it.tsv'))
    if minhash_dict and not ocsvm_it_file.is_file():
        print('Performing Anchored Recruitment with OC-SVM...')
        mh_trusted_df = minhash_dict[201]
        mh_trusted_df.rename(columns={'q_contig_id': 'contig_id'}, inplace=True)
        anchor_df = pd.read_csv(anchor_emb, sep='\t', header=0, index_col='subcontig_id')
        pool = multiprocessing.Pool(processes=nthreads)
        arg_list = []
        for sag_id in mh_trusted_df['sag_id'].unique():
            arg_list.append([anchor_df, mh_trusted_df, no_noit_df, noit_df, sag_id])
        ocsvm_recruit_list = []
        results = pool.imap_unordered(recruitOCSVM, arg_list)
        for i, output in tqdm(enumerate(results, 1)):
            if isinstance(output, pd.DataFrame):
                ocsvm_recruit_list.append(output)
        pool.close()
        pool.join()
        ocsvm_contig_df = pd.concat(ocsvm_recruit_list)
        ocsvm_contig_best_df = ocsvm_contig_df.sort_values(by='percent', ascending=False
                                                           ).drop_duplicates(subset='contig_id')
        sag_contig_list = []
        for index, row in tqdm(ocsvm_contig_best_df.iterrows()):
            sag_id = row['sag_id']
            percent = row['percent']
            sub_sag_contig_df = ocsvm_contig_df.query('sag_id == @sag_id and '
                                                      'percent >= @percent'
                                                      )
            sag_contig_list.append(sub_sag_contig_df)
        contig_pruned_df = pd.concat(sag_contig_list)

        print('Building OC-SVM Clusters...')
        ocsvm_clust_list = []
        for sag_id in tqdm(mh_trusted_df['sag_id'].unique()):
            trust_cols = ['sag_id', 'contig_id']
            sub_trusted_df = mh_trusted_df.query('sag_id == @sag_id and jacc_sim >= 1.0'
                                                 )[trust_cols]
            sub_ocsvm_df = contig_pruned_df.query('sag_id == @sag_id')[trust_cols]
            ocsvm_cat_df = pd.concat([sub_trusted_df, sub_ocsvm_df]).drop_duplicates()
            ocsvm_clust_list.append(ocsvm_cat_df)

        ocsvm_clust_df = pd.concat(ocsvm_clust_list)
        ocsvm_clust_df.rename(columns={'sag_id': 'best_label'}, inplace=True)
        ocsvm_clust_df.to_csv(ocsvm_it_file, sep='\t', index=False)
    elif ocsvm_it_file.is_file():
        print('OC-SVM Clusters already exist...')
        ocsvm_clust_df = pd.read_csv(ocsvm_it_file, sep='\t', header=0)
    else:
        ocsvm_clust_df = False

    # Find intersection of HDBSCAN and OC-SVM
    inter_it_file = Path(o_join(clst_path, mg_id + '.inter_it.tsv'))
    if minhash_dict and not inter_it_file.is_file():
        print('Combining Recruits from HDBSCAN and OC-SVM...')
        inter_clust_list = []
        for best_label in tqdm(trust_recruit_df['best_label'].unique()):
            sub_scan_df = trust_recruit_df.query('best_label == @best_label')
            sub_svm_df = ocsvm_clust_df.query('best_label == @best_label')
            intersect = set(sub_scan_df['contig_id']).intersection(set(sub_svm_df['contig_id']))
            best_inter_list = [(best_label, x) for x in intersect]
            best_inter_df = pd.DataFrame(best_inter_list, columns=['best_label', 'contig_id'])
            inter_clust_list.append(best_inter_df)
        inter_clust_df = pd.concat(inter_clust_list)
        inter_clust_df.to_csv(inter_it_file, sep='\t', index=False)
    elif inter_it_file.is_file():
        print('Combined Clusters already exist...')
        inter_clust_df = pd.read_csv(inter_it_file, sep='\t', header=0)
    else:
        inter_clust_df = False

    return no_noise_df, trust_recruit_df, ocsvm_clust_df, inter_clust_df
