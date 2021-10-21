import Colour_template as ct
from importlib import reload

reload(ct)

test_obj = ct.Colour_template()

test = test_obj.get_stimulus_names("Silent_Substitution", only_on=True)
test
test_test = np.asarray(test)

ticktext = test_obj.get_stimulus_names(
    "Silent_Substitution", only_on=True, only_off=False
)
ticktext


test[0]
[int(s) for s in test[0].split() if s.isdigit()]

import re

int(re.findall(r"\d+", "630nm")[0])


test_obj.list_stimuli()

import Opsins

reload(Opsins)

test = Opsins.Opsin_template()

import matplotlib.pyplot as plt

plt.plot(test.govardovskii_animal("Human"))
plt.show()


import numpy as np

test = np.array([1, 2, 3, 4, 5])
test_idx = np.array([1, 2])
test[test_idx]
