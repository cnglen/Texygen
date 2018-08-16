#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ifile = "./experiment-log-seqgan.csv"
    # ifile = "./experiment-log-mle.csv"
    raw_data = pd.read_csv(ifile)
    raw_data.plot(x="idx_epoch", y=["nll-oracle", "nll-test", "EmbeddingSimilarity"], grid=True, marker="o")
    plt.show()


if __name__ == '__main__':
    main()
