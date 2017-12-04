import argparse
import os

import numpy as np
from unet_model import create_model

CLASSES = 2
CHANNELS = 4
THRESHOLD = 0.5
BATCHSIZE = 16
EXPECTED_DIM = 2048
CROP_DIM = 256
OVERLAP = 0.5


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Process predicted image, creates geojson')
    arg_parser.add_argument('--model', required=True, help='Model to use')
    arg_parser.add_argument('--image', required=True, type=str,
                            help='Image to run prediction on or list of files')
    arg_parser.add_argument('--out-dir', required=True,
                            help='Directory to put predicted files')
    args = arg_parser.parse_args()
    if not os.path.isfile(args.model):
        raise RuntimeError('Cant open model %s' % args.model)
    if not os.path.isfile(args.image):
        raise RuntimeError('Cant open image %s' % args.image)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    return args


def crop_image(im):
    pieces = []
    start_pts = np.linspace(0, EXPECTED_DIM, EXPECTED_DIM / (OVERLAP * CROP_DIM) + 1)
    start_pts = start_pts[:-2]
    start_pts = [int(x) for x in start_pts]
    for i in start_pts:
        for j in start_pts:
            croped_image = im[i:i+CROP_DIM, j:j+CROP_DIM, :]
            pieces.append(croped_image)
    return pieces


def main():
    args = parse_args()
    todo = []
    if os.path.splitext(args.image)[1] != '.npy':
        with open(args.image, 'r') as infp:
            for line in infp:
                line = line.rstrip()
                todo.append(line)
    else:
        todo.append(args.image)
    model = create_model()
    model.load_weights(args.model)
    for path in todo:
        feat = np.load(path)
        dims = feat.shape
        assert(len(dims) == 3)
        print(dims[0])
        print(dims[1])
        assert(dims[0] == dims[1] == EXPECTED_DIM)
        assert(dims[2] == CHANNELS)
        feats_croped = crop_image(feat)
        num_feats = len(feats_croped)
        num_batches = int(np.ceil(num_feats / float(BATCHSIZE)))
        samples = num_batches * BATCHSIZE
        x = np.zeros((samples, CROP_DIM, CROP_DIM, CHANNELS))
        for i in range(num_feats):
            x[i, :, :, :] = feats_croped[i]
        y = model.predict(x, verbose=1, batch_size=BATCHSIZE)
        print(y.shape)
        for i in range(num_feats):
            yy = y[i, :, :]
            print(yy.shape)
            name = os.path.splitext(os.path.basename(path))[0]
            np.save(os.path.join(args.out_dir, '%s_%d' % (name, i)), yy)


if __name__ == '__main__':
    main()