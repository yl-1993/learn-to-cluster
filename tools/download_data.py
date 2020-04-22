import gdown
import tarfile
import argparse
import os.path as osp

data2url = {
    'part1':
    'https://drive.google.com/uc?id=16WD4orcF9dqjNPLzST2U3maDh2cpzxAY',
    'benchmark':
    'https://drive.google.com/uc?id=10boLBiYq-6wKC_N_71unlMyNrimRjpVa',
    'youtube_face':
    'https://drive.google.com/uc?id=1zrckFOx5fDnvDSK3ZeT2Di6HLaxZPnoG',
    'deepfashion':
    'https://drive.google.com/uc?id=15B5Ypj8_U9rhcuvkrkCZQAgV4cfes7aV',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download Dataset')
    parser.add_argument('--data', default='part1', choices=data2url.keys())
    parser.add_argument('--tar_path', default='data.tar.gz', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    url = data2url[args.data]
    tar_path = args.tar_path
    if osp.exists(tar_path) and not args.force:
        print('{} already exists.'
              'Modify --tar_path or run with --force to overwrite.'.format(
                  tar_path))
    else:
        gdown.download(url, tar_path, quiet=False)
    print('untar {}'.format(tar_path))
    tar = tarfile.open(tar_path)
    for member in tar.getmembers():
        if member.isreg():
            tar.extract(member)
    tar.close()
    print('download data successfully!')
