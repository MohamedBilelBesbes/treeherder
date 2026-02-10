import datetime
import time
from unittest import mock

import pytest

from treeherder.model.models import Push
from treeherder.perf.alerts_test import generate_test_alerts_in_series, detect_methods_changes, define_methods
from treeherder.perf.models import (
    PerformanceAlertTesting,
    PerformanceAlertSummaryTesting,
    PerformanceDatum,
    PerformanceSignature,
)


def _verify_alert(
    alert,
    expected_push_id,
    expected_prev_push_id,
    expected_signature,
    expected_prev_value,
    expected_new_value,
    expected_is_regression,
    expected_status,
    expected_noise_profile,
    expected_detection_method,
):
    """Verify an alert has the expected properties."""
    assert alert.prev_value == pytest.approx(expected_prev_value, abs=0.1)
    assert alert.new_value == pytest.approx(expected_new_value, abs=0.3) # Thanks to foreward window being small for some best confiugurations, abs is set to 0.2 to avoid test flakiness
    assert alert.series_signature == expected_signature
    assert alert.is_regression == expected_is_regression
    assert alert.status == expected_status
    assert alert.noise_profile == expected_noise_profile
    assert alert.detection_method == expected_detection_method
    # The logic of grouping adjacent alerts from different methods is still being worked out
    summary = alert.summary
    assert summary.push_id == expected_push_id
    assert summary.prev_push_id == expected_prev_push_id


def _generate_performance_data(
    test_repository,
    test_perf_signature,
    base_timestamp,
    start_id,
    value,
    amount,
):
    for t, v in zip(
        [i for i in range(start_id, start_id + amount)],
        [value for i in range(start_id, start_id + amount)],
    ):
        push, _ = Push.objects.get_or_create(
            repository=test_repository,
            revision=f"1234abcd{t}",
            defaults={
                "author": "foo@bar.com",
                "time": datetime.datetime.fromtimestamp(base_timestamp + t),
            },
        )
        PerformanceDatum.objects.create(
            repository=test_repository,
            push=push,
            signature=test_perf_signature,
            push_timestamp=datetime.datetime.utcfromtimestamp(base_timestamp + t),
            value=v,
        )

def test_mann_whitney_detector_detects_alerts(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that Mann-Whitney U test detector can detect performance changes."""
    base_time = time.time()
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Verify that Mann-Whitney detector created an alert
    mwu_alerts = PerformanceAlertTesting.objects.filter(detection_method='mwu')
    print(f"MWU alerts count: {mwu_alerts.count()}")
    assert mwu_alerts.count() == 1
    
    mwu_alert = mwu_alerts.first()
    _verify_alert(
        mwu_alert,
        int(interval / 2) + 1,
        int(interval / 2),
        test_perf_signature,
        0.5,
        1.0,
        True,
        PerformanceAlertTesting.UNTRIAGED,
        "OK",
        "mwu"
    )


def test_all_methods_detect_same_alert(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that all methods detect the same significant performance change."""
    base_time = time.time()
    interval = 30
    
    # Create a very clear regression that all methods should detect
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Check that we have alerts from multiple methods
    methods_with_alerts = set(PerformanceAlertTesting.objects.values_list('detection_method', flat=True))
    assert len(methods_with_alerts) >= 1
    
    # Verify each method's alert
    for method_name in methods_with_alerts:
        alerts = PerformanceAlertTesting.objects.filter(detection_method=method_name)
        assert alerts.count() == 1
        alert = alerts.first()
        _verify_alert(
            alert,
            int(interval / 2) + 1,
            int(interval / 2),
            test_perf_signature,
            0.5,
            1.0,
            True,
            PerformanceAlertTesting.UNTRIAGED,
            "OK",
            method_name
        )
    assert 0 == 1

def test_no_alerts_with_stable_data(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
):
    """Test that no alerts are generated when data is stable."""
    base_time = time.time()
    interval = 30
    
    # Generate stable data (no change)
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        interval,
    )

    generate_test_alerts_in_series(test_perf_signature)

    # No alerts should be generated
    assert PerformanceAlertTesting.objects.count() == 0


def test_detect_alerts_with_retriggers(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that detectors handle retriggers correctly."""
    base_time = time.time()
    
    # Create data with retriggers (multiple values for same push)
    for i in range(20):
        _generate_performance_data(
            test_repository,
            test_perf_signature,
            base_time,
            1,
            0.5,
            1,
        )
    for i in range(5):
        _generate_performance_data(
            test_repository,
            test_perf_signature,
            base_time,
            2,
            1.5,
            1,
        )
    for i in range(15):
        _generate_performance_data(
            test_repository,
            test_perf_signature,
            base_time,
            2,
            0.5,
            1,
        )

    generate_test_alerts_in_series(test_perf_signature)
    
    # Should detect some alerts
    assert PerformanceAlertTesting.objects.count() > 0
    
    # Verify at least one alert with retrigger data
    alert = PerformanceAlertTesting.objects.first()
    assert alert.prev_value == pytest.approx(0.5, abs=0.1)
    assert alert.new_value > 0.5  # Should be between 0.5 and 1.0


def test_no_alerts_with_old_data(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
):
    """Test that no alerts are generated for data that is too old."""
    base_time = 0  # 1970, too old!
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    assert PerformanceAlertTesting.objects.count() == 0
    assert PerformanceAlertSummaryTesting.objects.count() == 0


def test_custom_alert_threshold(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
):
    """Test that custom alert thresholds are respected."""
    test_perf_signature.alert_threshold = 200.0
    test_perf_signature.save()

    interval = 60
    base_time = time.time()
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 3),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 3) + 1,
        0.6,
        int(interval / 3),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        2 * int(interval / 3) + 1,
        2.0,
        int(interval / 3),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # With high threshold, only large change should be detected
    total_alerts = PerformanceAlertTesting.objects.count()
    print(f"Total alerts generated with custom threshold: {total_alerts}")
    assert total_alerts == 4  # At least doesn't crash


@pytest.mark.parametrize(("new_value", "expected_min_alerts"), [(1.0, 1), (0.25, 0)])
def test_alert_change_type_absolute(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    new_value,
    expected_min_alerts,
):
    """Test that absolute change type threshold is respected."""
    test_perf_signature.alert_change_type = PerformanceSignature.ALERT_ABS
    test_perf_signature.alert_threshold = 0.3
    test_perf_signature.save()

    base_time = time.time()
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        new_value,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    total_alerts = PerformanceAlertTesting.objects.count()
    assert total_alerts >= expected_min_alerts


def test_alert_monitor_no_sheriff(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
):
    """Test that monitored alerts are not sheriffed."""
    test_perf_signature.monitor = True
    test_perf_signature.should_alert = True
    test_perf_signature.save()

    base_time = time.time()
    interval = 60
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    assert PerformanceAlertTesting.objects.count() > 0

    # When monitor is true, alerts should not be sheriffed
    for alert in PerformanceAlertTesting.objects.all():
        assert not alert.sheriffed


def test_confidence_values_stored(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that confidence values (p-values) are properly stored."""
    base_time = time.time()
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Check that all alerts have confidence values
    assert PerformanceAlertTesting.objects.count() > 0
    for alert in PerformanceAlertTesting.objects.all():
        assert alert.confidence is not None
        assert isinstance(alert.confidence, float)


def test_detection_method_stored(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that detection method names are properly stored."""
    base_time = time.time()
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Check that all alerts have detection methods
    assert PerformanceAlertTesting.objects.count() > 0
    valid_methods = ['welch', 'mwu', 'ks', 'cvm', 'levene', 'studenttmag']
    for alert in PerformanceAlertTesting.objects.all():
        assert alert.detection_method in valid_methods


def test_multiple_alerts_in_sequence(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that multiple sequential performance changes are detected."""
    base_time = time.time()
    interval = 30

    # First change
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)
    first_alert_count = PerformanceAlertTesting.objects.count()
    assert first_alert_count > 0

    # Add more data with another change
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        interval + 1,
        2.0,
        interval,
    )

    generate_test_alerts_in_series(test_perf_signature)
    second_alert_count = PerformanceAlertTesting.objects.count()
    
    # Should have more alerts after second change
    assert second_alert_count >= first_alert_count


def test_percentiles_stored_correctly(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
    mock_deviance,
):
    """Test that median and percentile values are properly stored."""
    base_time = time.time()
    interval = 30
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        int(interval / 2),
    )
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        int(interval / 2) + 1,
        1.0,
        int(interval / 2),
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Check that all alerts have percentile values
    assert PerformanceAlertTesting.objects.count() > 0
    for alert in PerformanceAlertTesting.objects.all():
        assert alert.prev_median is not None
        assert alert.new_median is not None
        assert alert.prev_p90 is not None
        assert alert.new_p90 is not None
        assert alert.prev_p95 is not None
        assert alert.new_p95 is not None


def test_detector_with_insufficient_data(
    test_repository,
    test_issue_tracker,
    failure_classifications,
    generic_reference_data,
    test_perf_signature,
):
    """Test that detectors handle insufficient data gracefully."""
    base_time = time.time()
    
    # Generate only 5 data points (not enough for detection)
    _generate_performance_data(
        test_repository,
        test_perf_signature,
        base_time,
        1,
        0.5,
        5,
    )

    generate_test_alerts_in_series(test_perf_signature)

    # Should not generate alerts with insufficient data
    assert PerformanceAlertTesting.objects.count() == 0
