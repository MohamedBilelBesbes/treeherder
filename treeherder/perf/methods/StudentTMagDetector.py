import copy
import functools
from methods import BaseDetector

class StudentTMagDetector(BaseDetector):

    def calc_alpha(self, w1, w2, weight_fn=None):
        """Perform a Students t-test on the two sets of revision data.

        See the analyze() function for a description of the `weight_fn` argument.
        """
        if not w1 or not w2:
            return 0

        s1 = self.analyze(w1, weight_fn)
        s2 = self.analyze(w2, weight_fn)
        delta_s = s2["avg"] - s1["avg"]

        if delta_s == 0:
            return 0
        if s1["variance"] == 0 and s2["variance"] == 0:
            return float("inf")

        return delta_s / (((s1["variance"] / s1["n"]) + (s2["variance"] / s2["n"])) ** 0.5)