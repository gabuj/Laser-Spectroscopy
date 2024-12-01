import scipy.constants as const
import numpy as numpy

#define constants

c= const.c
h= const.h
e= const.e

# Transition frequencies for Rubidium 87 in Hz
Rb85S2_P1 = 384228522.3211e6
Rb85S2_P2 = 384228551.6941e6
Rb85S2_P3 = 384228615.0941e6
Rb85S3_P2 = 384229057.6495e6
Rb85S3_P3 = 384229121.0495e6
Rb85S3_P4 = 384229241.6895e6

Rb85S2_P1_err = 0.0852e6
Rb85S2_P2_err = 0.0368e6
Rb85S2_P3_err = 0.0529e6
Rb85S3_P2_err = 0.0368e6
Rb85S3_P3_err = 0.0529e6
Rb85S3_P4_err = 0.0462e6


# Transition frequencies for Rubidium 87 in Hz

Rb87S1_P0 = 384225910.718068e6  # F=1 in S to F=0 in P
Rb87S1_P1 = 384226003.940068e6  # F=1 in S to F=1 in P
Rb87S1_P2 = 384226140.880668e6  # F=1 in S to F=2 in P

Rb87S2_P1 = 384227691.610721e6  # F=2 in S to F=1 in P
Rb87S2_P2 = 384227848.551321e6  # F=2 in S to F=2 in P
Rb87S2_P3 = 384227727.721821e6  # F=2 in S to F=3 in P

#Uncertainties
Rb87S1_P0_err = 0.0108e6  # F=1 in S to F=0 in P
Rb87S1_P1_err = 0.0084e6  # F=1 in S to F=1 in P
Rb87S1_P2_err = 0.0070e6  # F=1 in S to F=2 in P

Rb87S2_P1_err = 0.0084e6  # F=2 in S to F=1 in P
Rb87S2_P2_err = 0.0070e6  # F=2 in S to F=2 in P
Rb87S2_P3_err = 0.0077e6  # F=2 in S to F=3 in P


# Old

# Rb87F1_P_f= Rb87S_F_f + 4.27167663181518
# Rb87F1_P_f_err= Rb87F1_P_f 