from scipy import stats

from treeherder.perf.methods.BaseDetector import BaseDetector


class CramerVonMisesDetector(BaseDetector):
    """
    Detector using Cramér-von Mises test.
    """

    def __init__(
        self,
        min_back_window=12,
        max_back_window=24,
        fore_window=12,
        magnitude_threshold=2.0,
        alpha_threshold=0.05,
        mag_check=False,
        above_threshold_is_anomaly=False,
    ):
        super().__init__(
            min_back_window=min_back_window,
            max_back_window=max_back_window,
            fore_window=fore_window,
            magnitude_threshold=magnitude_threshold,
            alpha_threshold=alpha_threshold,
            mag_check=mag_check,
            above_threshold_is_anomaly=above_threshold_is_anomaly,
        )

    def calc_alpha(self, jw, kw, alpha_threshold, confidence):
        """
        Calculate Cramér-von Mises test statistic and p-value.
        """
        jw_values = [v for datum in jw for v in datum.values]
        kw_values = [v for datum in kw for v in datum.values]

        if len(jw_values) < 2 or len(kw_values) < 2:
            return 1.0, confidence + 1

        try:
            result = stats.cramervonmises_2samp(jw_values, kw_values)
            p = result.pvalue
        except Exception:
            p = 1.0

        if p < alpha_threshold:
            confidence = 0
        else:
            confidence += 1

        return p, confidence
