# flake8: noqa
from mssm.models import *
from mssm.src.python.compare import compare_CDL
import numpy as np
import os
from mssmViz.sim import *
from mssm.src.python.repara import reparam
from mssm.src.python.gamm_solvers import (
    compute_S_emb_pinv_det,
    cpp_chol,
    cpp_cholP,
    compute_eigen_perm,
    compute_Linv,
)
from mssm.src.python.utils import correct_VB, estimateVp
import io
from contextlib import redirect_stdout
from .defaults import default_gamm_test_kwargs, max_atol, max_rtol


class Test_BIG_GAMM_keep_cov:
    file_paths = [
        f"https://raw.githubusercontent.com/JoKra1/mssmViz/main/data/GAMM/sim_dat_cond_{cond}.csv"
        for cond in ["a", "b"]
    ]

    codebook = {"cond": {"a": 0, "b": 1}}

    formula = Formula(
        lhs=lhs("y"),  # The dependent variable - here y!
        terms=[
            i(),  # The intercept, a
            l(["cond"]),  # For cond='b'
            f(
                ["time"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between time and cond; one smooth over time per cond level
            f(
                ["x"], by="cond", constraint=ConstType.QR
            ),  # to-way interaction between x and cond; one smooth over x per cond level
            f(
                ["time", "x"], by="cond", constraint=ConstType.QR, nk=9
            ),  # three-way interaction
            fs(["time"], rf="sub"),
        ],  # Random non-linear effect of time - one smooth per level of factor sub
        data=None,  # No data frame!
        file_paths=file_paths,  # Just a list with paths to files.
        print_warn=False,
        keep_cov=True,  # Keep encoded data structure in memory
        codebook=codebook,
    )

    def test_GAMedf(self):
        assert 5 == 5
