from scipy import stats

from treeherder.perf.methods.BaseDetector import BaseDetector


class WelchDetector(BaseDetector):
    """
    Detector using Welch's t-test (unequal variances t-test).
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

    def calc_confidence(self, jw, kw, confidence_threshold, last_seen_regression):
        """
        Calculate Welch's t-test statistic and p-value.
        """
        jw_values = [v for datum in jw for v in datum.values]
        kw_values = [v for datum in kw for v in datum.values]

        if len(jw_values) < 2 or len(kw_values) < 2:
            return 1.0, last_seen_regression + 1  # p-value of 1.0 (no significance)

        try:
            result = stats.ttest_ind(jw_values, kw_values, equal_var=False)
            p = result.pvalue
        except Exception:
            p = 1.0

        if p < confidence_threshold:
            last_seen_regression = 0
        else:
            last_seen_regression += 1

        return p, last_seen_regression
