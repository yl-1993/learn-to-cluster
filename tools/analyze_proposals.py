import os
import glob
from utils import load_data
from tqdm import tqdm

if __name__ == '__main__':
    fn_node_pattern = '*_node.npz'
    folder_pattern = './data/cluster_proposals/part1_test/{}/proposals/'
    proposal_folders = [
        folder_pattern.format(
            'faiss_k_30_th_0.7_step_0.05_minsz_3_maxsz_300_iter_0'),
        folder_pattern.format(
            'faiss_k_80_th_0.7_step_0.05_minsz_3_maxsz_300_iter_0')
    ]
    nodes = []
    for proposal_folder in tqdm(proposal_folders):
        fn_nodes = sorted(
            glob.glob(os.path.join(proposal_folder, fn_node_pattern)))
        nodes.append([set(load_data(fn_node)) for fn_node in fn_nodes])

    cnt = 0
    print(len(nodes[0]), len(nodes[1]))
    for n1 in tqdm(nodes[0]):
        for n2 in tqdm(nodes[1]):
            inter = len(n1 & n2)
            # if inter > 0:
            if inter > 0 and inter < len(n1) and inter < len(n2):
                cnt += 1
                print('cnt:', cnt, inter, len(n1), len(n2))
                # break
                # exit()
    print('cnt:', cnt)
