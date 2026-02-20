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
        confidence_threshold=0.05,
        mag_check=False,
        above_threshold_is_anomaly=False,
    ):
        super().__init__(
            min_back_window=min_back_window,
            max_back_window=max_back_window,
            fore_window=fore_window,
            magnitude_threshold=magnitude_threshold,
            confidence_threshold=confidence_threshold,
            mag_check=mag_check,
            above_threshold_is_anomaly=above_threshold_is_anomaly,
        )

    def calc_confidence(
        self, jw, kw, confidence_threshold, last_seen_regression, replicates_enabled
    ):
        """
        Calculate Cramér-von Mises test statistic and p-value.
        """
        source_attr = "replicates" if replicates_enabled else "values"

        jw_values = [v for datum in jw for v in getattr(datum, source_attr)]
        kw_values = [v for datum in kw for v in getattr(datum, source_attr)]

        if len(jw_values) < 2 or len(kw_values) < 2:
            return 1.0, last_seen_regression + 1

        try:
            result = stats.cramervonmises_2samp(jw_values, kw_values)
            p = result.pvalue
        except Exception:
            p = 1.0

        if p < confidence_threshold:
            last_seen_regression = 0
        else:
            last_seen_regression += 1

        return p, last_seen_regression
