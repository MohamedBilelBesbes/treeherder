import BaseDetector
from scipy import stats

class LeveneDetector(BaseDetector):
    """
    Detector using Levene's test (tests for equal variances).
    """
    
    def __init__(self, min_back_window=12, max_back_window=24, fore_window=12,
                 alert_threshold=2.0, alpha_threshold=0.05, mag_check=False):
        self.min_back_window = min_back_window
        self.max_back_window = max_back_window
        self.fore_window = fore_window
        self.alert_threshold = alert_threshold
        self.alpha_threshold = alpha_threshold
        self.mag_check = mag_check
        super().__init__()
    
    def calc_alpha(self, jw, kw, alpha_threshold, last_seen_regression, weight_fn):
        """
        Calculate Levene's test statistic and p-value.
        """
        jw_values = [v for datum in jw for v in datum.values]
        kw_values = [v for datum in kw for v in datum.values]
        
        if len(jw_values) < 2 or len(kw_values) < 2:
            return 1.0, last_seen_regression + 1
        
        try:
            stat, p = stats.levene(jw_values, kw_values)
        except Exception:
            stat, p = 0, 1.0
        
        if p < alpha_threshold:
            last_seen_regression = 0
        else:
            last_seen_regression += 1
        
        return p, last_seen_regression
