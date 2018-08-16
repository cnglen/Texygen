#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ifile = "./experiment-log-seqgan.csv"
    # ifile = "./experiment-log-mle.csv"
    raw_data = pd.read_csv(ifile)

    for icol, col in enumerate(["nll-oracle", "nll-test", "EmbeddingSimilarity"]):
        plt.subplot(3, 1, icol + 1)
        plt.plot(raw_data.idx_epoch, raw_data[col], marker="o")
        plt.xlabel("idx_epoch")
        plt.ylabel(col)
        plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
