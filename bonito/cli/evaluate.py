"""
Bonito model evaluator
"""

import os
import time
import torch
import numpy as np
from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.data import load_numpy, load_script
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_model, concat, permute

from torch.utils.data import DataLoader


def main(args):

    poas = []
    init(args.seed, args.device)

    print("* loading data")
    try:
        _, valid_loader_kwargs = load_numpy(args.directory, args.chunks, load_training_set=False)
    except FileNotFoundError:
        _, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.chunks
        )

    dataloader = DataLoader(
        batch_size=args.batchsize, num_workers=4, pin_memory=True,
        **valid_loader_kwargs
    )

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=args.min_coverage)

    for w in [int(i) for i in args.weights.split(',')]:

        seqs = []

        print("* loading model", w)
        model = load_model(args.model_directory, args.device, decode_method=args.decode_method, weights=w)

        print("* calling")
        t0 = time.perf_counter()

        targets = []
        if(args.save_DNN_output):
            saveDir = os.path.join(args.directory,'../DNNOutput',os.path.basename(args.model_directory))
            os.makedirs(saveDir, exist_ok=True)
        with torch.no_grad():
            for (i,(data, target, *_)) in enumerate(dataloader):
                targets.extend(torch.unbind(target, 0))
                if not args.load_DNN_output: # Standard mode: run the DNN with the validation data
                    if half_supported():
                        data = data.type(torch.float16).to(args.device)
                    else:
                        data = data.to(args.device)

                    log_probs = model(data)
                    if(args.save_DNN_output): # Save DNN outputs so DNN doesn't have to be called every time we want to run decoding
                        torch.save(log_probs.cpu(),os.path.join(saveDir,f'output_{i}.tensor'))
                else: # Fast mode: Only run the decoding with the presaved values
                    loadDir = os.path.join(args.directory,'../DNNOutput',os.path.basename(args.model_directory))
                    log_probs = torch.load(os.path.join(loadDir, f'output_{i}.tensor'), map_location=args.device)
                
                if hasattr(model, 'decode_batch'):
                    seqs.extend(model.decode_batch(log_probs)) # <---- Here is where decoding is called
                else:
                    seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])
        duration = time.perf_counter() - t0

        refs = [decode_ref(target, model.alphabet) for target in targets]
        accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

        if args.poa: poas.append(sequences)

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)
        print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))

    if args.poa:

        print("* doing poa")
        t0 = time.perf_counter()
        # group each sequence prediction per model together
        poas = [list(seq) for seq in zip(*poas)]
        consensuses = poa(poas)
        duration = time.perf_counter() - t0
        accuracies = list(starmap(accuracy_with_coverage_filter, zip(references, consensuses)))

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=1000, type=int)
    parser.add_argument("--batchsize", default=96, type=int)
    parser.add_argument("--save_DNN_output", action="store_true", default=False)
    parser.add_argument("--load_DNN_output", action="store_true", default=False)
    parser.add_argument("--decode_method",default="ont")
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser

if __name__ == '__main__':
    # Execute only if run as the entry point into the program
    parser = argparser()
    m_args = parser.parse_args()
    main(m_args)