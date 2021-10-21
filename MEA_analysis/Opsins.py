import numpy as np


class Opsin_template:
    opsins_peaks = {}
    opsins_peaks["Chicken"] = np.array([570, 508, 455, 419])
    opsins_peaks["Zebrafish"] = np.array([548, 476, 416, 355])
    opsins_peaks["Human"] = np.array([564, 534, 420])

    connumbers = {}
    connumbers["Chicken"] = int(4)
    connumbers["Zebrafish"] = int(4)
    connumbers["Human"] = int(3)

    def govardovskii_animal(self, animal_name):

        return govardovskii(
            self.opsins_peaks[animal_name], self.connumbers[animal_name]
        )


def govardovskii(peakwvss, conenum):
    Ops = np.zeros((1025, conenum))
    for gg in range(0, conenum):
        awvs = 0.8795 + 0.0459 * np.exp(-1 * (peakwvss[gg] - 300) ** 2 / 11940)
        constA = np.asarray([awvs, 0.922, 1.104, 69.7, 28, -14.9, 0.674, peakwvss[gg]])
        Lamb = 189 + 0.315 * constA[7]  # Beta peak
        b = -40.5 + 0.195 * constA[7]  # Beta bandwidth
        Ab = 0.26  # Beta value at peak
        constB = np.asarray([Ab, Lamb, b])
        awvs = 0.8795 + 0.0459 * np.exp(-1 * (constA[7] - 300) ** 2 / 11940)
        constAB = np.concatenate([constA[:], constB[:]])

        lalax = np.asarray(list(range(300, 700, 1)))
        lalay = temple(lalax, constAB)  # Govardovskii guess

        Ops[300:700:1, gg] = lalay
    return Ops[300:700, :]


def temple(x, constAB):
    # constAB = a,b,c,A,B,C,D,lamA,Ab,Lamb,bB
    constA = constAB[0:8:1]
    constB = constAB[8:11:1]
    S = alphaband(x, constA) + betaband(x, constB)
    return S


def alphaband(x, constA):
    # a,b,c,A,B,C,D,lamA = constA
    x = constA[7] / x
    ##A = b/(a+b)*numpy.exp(a/n)
    ##B = a/(a+b)*numpy.exp(-1*b/n)
    alpha = 1 / (
        np.exp(constA[3] * (constA[0] - x))
        + np.exp(constA[4] * (constA[1] - x))
        + np.exp(constA[5] * (constA[2] - x))
        + constA[6]
    )
    return alpha


def betaband(x, constB):
    beta = constB[0] * np.exp(-1 * ((x - constB[1]) / constB[2]) ** 2)
    return beta
