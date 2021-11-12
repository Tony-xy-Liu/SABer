__author__ = 'Ryan J McLaughlin'

import glob
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from itertools import product, islice

import pandas as pd
import pyfastx
from skbio.stats.composition import clr
from sklearn.preprocessing import StandardScaler


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path_element in os.environ["PATH"].split(os.pathsep):
            path_element = path_element.strip('"')
            exe_file = os.path.join(path_element, program)
            if is_exe(exe_file):
                return exe_file
    return None


def executable_dependency_versions(exe_dict):
    """Function for retrieving the version numbers for each executable in exe_dict
    :param exe_dict: A dictionary mapping names of software to the path to their executable
    :return: A formatted string with the executable name and its respective version found"""
    versions_dict = dict()
    versions_string = "Software versions used:\n"

    simple_v = ["prodigal"]
    no_params = ["bwa"]
    version_re = re.compile(r"[Vv]\d+.\d|version \d+.\d|\d\.\d\.\d")

    for exe in exe_dict:
        ##
        # Get the help/version statement for the software
        ##
        versions_dict[exe] = ""
        if exe in simple_v:
            stdout, returncode = launch_write_command([exe_dict[exe], "-v"], True)
        elif exe in no_params:
            stdout, returncode = launch_write_command([exe_dict[exe]], True)
        else:
            logging.warning("Unknown version command for " + exe + ".\n")
            continue
        ##
        # Identify the line with the version number (since often more than a single line is returned)
        ##
        for line in stdout.split("\n"):
            if version_re.search(line):
                # If a line was identified, try to get just the string with the version number
                for word in line.split(" "):
                    if re.search(r"\d\.\d", word):
                        versions_dict[exe] = re.sub(r"[,:()[\]]", '', word)
                        break
                break
            else:
                pass
        if not versions_dict[exe]:
            logging.debug("Unable to find version for " + exe + ".\n")

    ##
    # Format the string with the versions of all software
    ##
    for exe in sorted(versions_dict):
        n_spaces = 12 - len(exe)
        versions_string += "\t" + exe + ' ' * n_spaces + versions_dict[exe] + "\n"

    return versions_string


def launch_write_command(cmd_list, just_do_it=False, collect_all=True):
    """Wrapper function for opening subprocesses through subprocess.Popen()

    :param cmd_list: A list of strings forming a complete command call
    :param just_do_it: Always return even if the returncode isn't 0
    :param collect_all: A flag determining whether stdout and stderr are returned
    via stdout or just stderr is returned leaving stdout to be written to the screen
    :return: A string with stdout and/or stderr text and the returncode of the executable"""
    stdout = ""
    if collect_all:
        proc = subprocess.Popen(cmd_list,
                                shell=False,
                                preexec_fn=os.setsid,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        stdout = proc.communicate()[0].decode("utf-8")
    else:
        proc = subprocess.Popen(cmd_list,
                                shell=False,
                                preexec_fn=os.setsid)
        proc.wait()

    # Ensure the command completed successfully
    if proc.returncode != 0 and not just_do_it:
        logging.error(cmd_list[0] + " did not complete successfully! Command used:\n" +
                      ' '.join(cmd_list) + "\nOutput:\n" + stdout)
        sys.exit(19)

    return stdout, proc.returncode


def check_out_dirs(save_path):
    """Checks if dirs all exist in save_path, makes them if not.

    :param save_path: directory where all intermediate and final files are saved.
    :return: A dictionary with the stage dir and the full path."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sd_list = ['tmp', 'xPGs']
    sd_dict = {}
    for sd in sd_list:
        sd_path = os.path.join(save_path, sd)
        if not os.path.exists(sd_path):
            os.makedirs(sd_path)
        sd_dict[sd] = sd_path

    return sd_dict


def get_SAGs(sag_path):
    # Find the SAGs!
    if os.path.isdir(sag_path):
        logging.info('Directory specified, looking for Trusted Contigs\n')
        sag_list = [os.path.join(sag_path, f) for f in
                    os.listdir(sag_path) if ((f.split('.')[-1] == 'fasta' or
                                              f.split('.')[-1] == 'fna' or
                                              f.split('.')[-1] == 'fa') and 'Sample' not in f)
                    ]
        logging.info('Found %s Trusted Contig files in directory\n'
                     % str(len(sag_list))
                     )

    elif os.path.isfile(sag_path):
        logging.info('File specified, processing %s\n'
                     % os.path.basename(sag_path)
                     )
        sag_list = [sag_path]

    return sag_list


def build_subcontigs(seq_type, in_fasta_list, subcontig_path, max_contig_len, overlap_len):
    sub_list = []
    for i, in_fasta in enumerate(in_fasta_list):
        basename = os.path.basename(in_fasta)
        samp_id = basename.rsplit('.', 1)[0]
        sub_file = os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')
        logging.info('\rLoading/Building subcontigs for {}: {}'.format(seq_type, i + 1))
        if os.path.exists(os.path.join(subcontig_path, samp_id + '.subcontigs.fasta')) == False:
            # get contigs from fasta file
            contigs = get_seqs(in_fasta)
            headers, subs = kmer_slide(contigs, int(max_contig_len),
                                       int(overlap_len)
                                       )
            if len(subs) != 0:
                with open(sub_file, 'w') as sub_out:
                    sub_out.write('\n'.join(['\n'.join(['>' + rec[0], rec[1]]) for rec in
                                             zip(headers, subs)]) + '\n'
                                  )
                sub_list.append((samp_id, sub_file))
        else:
            sub_list.append((samp_id, sub_file))

    logging.info('\n')
    if ((seq_type == 'SAGs') & (len(sub_list) == 1)):
        sub_list = tuple(sub_list)
        return sub_list
    elif len(sub_list) == 1:
        sub_file = sub_list[0]
        return sub_file
    else:
        sub_list = tuple(sub_list)
        return sub_list


def kmer_slide(scd_db, n, o_lap):
    all_sub_seqs = []
    all_sub_headers = []
    for k in scd_db:
        rec = k
        header, seq = rec.name, rec.seq
        if len(str(seq)) >= int(o_lap):
            clean_seq = str(seq).upper()
            sub_list = sliding_window(clean_seq, n, o_lap)
            sub_headers = [header + '_' + str(i) for i, x in
                           enumerate(sub_list, start=0)
                           ]
            all_sub_seqs.extend(sub_list)
            all_sub_headers.extend(sub_headers)
        else:
            all_sub_seqs.extend([seq])
            all_sub_headers.extend([header + '_0'])
    return tuple(all_sub_headers), tuple(all_sub_seqs)


def sliding_window(seq, win_size, o_lap):
    "Fragments the seq into subseqs of length win_size and overlap of o_lap."
    "Leftover tail overlaps with tail-1"
    "Currently, if a seq is < win_size, it returns the full seq"
    seq_frags = []
    # Verify the inputs
    try:
        it = iter(seq)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(win_size) == type(0)) and (type(o_lap) == type(0))):
        raise Exception("**ERROR** type(win_size) and type(win_size) must be int.")
    if o_lap > win_size:
        raise Exception("**ERROR** step must not be larger than win_size.")
    if win_size <= len(seq):
        i = 0
        offset = len(seq) - win_size
        while i + win_size <= offset:
            seq_frags.append(seq[i:i + win_size])
            i = i + win_size - o_lap
        seq_frags.append(seq[-win_size:])
    elif win_size > len(seq):
        seq_frags.append(seq)

    return seq_frags


def slidingWindow(sequence, winSize, step):
    # pulled source from https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
    seq_frags = []
    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize <= len(sequence):
        numOfChunks = ((len(sequence) - winSize) // step) + 1
        for i in range(0, numOfChunks * step, step):
            seq_frags.append(sequence[i:i + winSize])
        seq_frags.append(sequence[-winSize:])  # add the remaining tail
    elif winSize > len(sequence):
        seq_frags.append(sequence)

    return seq_frags


def get_seqs(fasta_file):
    fasta = pyfastx.Fasta(fasta_file)

    return fasta


def get_kmer(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def tetra_cnt(fasta):  # TODO: add multi-processing to this function
    # Dict of all tetramers
    tetra_cnt_dict = {''.join(x): [] for x in product('atgc', repeat=4)}
    header_list = []
    # count up all tetramers and also populate the tetra dict
    subcontig_len_dict = {}
    for rec in fasta:
        header = rec.name
        header_list.append(header)
        seq = rec.seq
        subcontig_len_dict[header] = len(seq)
        tmp_dict = {k: 0 for k, v in tetra_cnt_dict.items()}
        clean_seq = seq.strip('\n').lower()
        kmer_list = [''.join(x) for x in get_kmer(clean_seq, 4)]
        tetra_counter = Counter(kmer_list)
        # add counter to tmp_dict
        for tetra in tmp_dict.keys():
            count_tetra = int(tetra_counter[tetra])
            tmp_dict[tetra] = count_tetra
        # map tetras to their reverse tetras (not compliment)
        dedup_dict = {}
        for tetra in tmp_dict.keys():
            if (tetra not in dedup_dict.keys()) & (tetra[::-1] not in dedup_dict.keys()):
                dedup_dict[tetra] = ''
            elif tetra[::-1] in dedup_dict.keys():
                dedup_dict[tetra[::-1]] = tetra
        # combine the tetras and their reverse (not compliment)
        tetra_prop_dict = {}
        for tetra in dedup_dict.keys():
            if dedup_dict[tetra] != '':
                tetra_prop_dict[tetra] = tmp_dict[tetra] + tmp_dict[dedup_dict[tetra]]
            else:
                tetra_prop_dict[tetra] = tmp_dict[tetra]
        # add to tetra_cnt_dict
        for k in tetra_cnt_dict.keys():
            if k in tetra_prop_dict.keys():
                tetra_cnt_dict[k].append(tetra_prop_dict[k])
            else:
                tetra_cnt_dict[k].append(0.0)
    # convert the final dict into a pd dataframe for ease
    tetra_cnt_dict['contig_id'] = header_list
    tetra_cnt_df = pd.DataFrame.from_dict(tetra_cnt_dict).set_index('contig_id')
    dedupped_df = tetra_cnt_df.loc[:, (tetra_cnt_df != 0.0).any(axis=0)]
    dedupped_df += 1  # TODO: adds pseudo-count, is there a better way?
    first_val = dedupped_df.columns[0]
    last_val = dedupped_df.columns[-1]
    dedupped_df['sum'] = dedupped_df.sum(axis=1)
    # Covert to proportion
    prop_df = dedupped_df.loc[:, first_val:last_val].div(dedupped_df['sum'], axis=0)
    # Normalize proportions to length of subcontig
    normal_list = [prop_df.loc[i] / subcontig_len_dict[i] for i in subcontig_len_dict.keys()]
    normal_df = pd.DataFrame(normal_list, columns=prop_df.columns, index=prop_df.index)
    # Transform using CLR
    clr_df = normal_df.apply(clr)
    # Standardize the mg tetra DF
    scale = StandardScaler().fit(clr_df.values)  # TODO this should be added to the tetra_cnt step
    scaled_data = scale.transform(clr_df.values)
    std_tetra_df = pd.DataFrame(scaled_data, index=clr_df.index)

    return std_tetra_df


def runCleaner(dir_path, ptrn):
    ptrn_glob = glob.glob(os.path.join(dir_path, ptrn))
    for ent in ptrn_glob:
        if os.path.isfile(ent):
            try:
                os.remove(ent)
            except:
                print("Error while deleting file : ", ent)
        elif os.path.isdir(ent):
            try:
                os.rmdir(ent)
            except:
                print("Error while deleting file : ", ent)


def set_clust_params(denovo_min_clust, denovo_min_samp, anchor_min_clust,
                     anchor_min_samp, nu, gamma, vr, r, s, vs
                     ):
    params_tmp = [denovo_min_clust, denovo_min_samp, anchor_min_clust,
                  anchor_min_samp, nu, gamma
                  ]

    cust_params = False
    params_list = [75.0, 10.0, 125.0, 10.0, 0.3, 0.1]  # TODO: should probs have a config file for all of these presets
    for i, p in enumerate(params_tmp):
        if p is not None:
            params_list[i] = float(p)
            cust_params = True
    if vr:
        params_list = [50.0, 5.0, 75.0, 10.0, 0.7, 10.0]
        cust_params = True
    elif r:
        params_list = [50.0, 10.0, 75.0, 10.0, 0.7, 10.0]
        cust_params = True
    elif s:
        params_list = [75.0, 10.0, 125.0, 10.0, 0.3, 0.1]
        cust_params = True
    elif vs:
        params_list = [75.0, 10.0, 125.0, 5.0, 0.3, 0.1]
        cust_params = True
    if not cust_params:  # No custom params have been set, proceed with automatic param algo
        params_list = auto_params()

    return params_list
