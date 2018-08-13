import getopt
import sys

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd
import argparse


def set_gan(gan_name):
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


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:t:d:")

        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t real -d <your_data_location>')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                gan_func(opt_arg['-d'])
            else:
                gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="app_texygen", description="using texygen to train a generator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-g", "--gan-type", default="mle", help="seqgan|gsgan|textgan|leakgan|rankgan|maligan|mle")
    parser.add_argument("-t", "--train-type", default="oracle", help="oracle|cfg|real")
    parser.add_argument("-d", "--data-location", default=None, help="data location")

    args = parser.parse_args()

    gan = set_gan(args.gan_type)
    gan_func = set_training(gan, args.train_type)
    if args.train_type == "real":
        gan_func(args.data_location)
