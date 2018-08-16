#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import argparse
from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


def set_gan(gan_name):
    """get a Gan instance of type gan_name"""
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan, training_method):
    """get a training function for `gan` using `training_method`"""
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="app_texygen",
                                     description="using texygen to train a generator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gan-type",
                        default="mle",
                        help="type of gan",
                        choices=["seqgan", "gsgan", "textgan", "leakgan", "rankgan", "maligan", "mle"])
    parser.add_argument("-t", "--train-type",
                        default="oracle",
                        help="type of training",
                        choices=["oracle", "cfg", "real"])
    parser.add_argument("-d", "--data-location",
                        default=None,
                        help="data location")
    args = parser.parse_args()

    gan = set_gan(args.gan_type)
    gan_func = set_training(gan, args.train_type)
    if args.train_type == "real":
        gan_func(args.data_location)
    else:
        gan_func()
