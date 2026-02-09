import logging
import time
from collections import namedtuple
from datetime import datetime

import moz_measure_noise
import newrelic.agent
import numpy as np
from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef, Subquery
import copy

from treeherder.perf.email import AlertNotificationWriter
from treeherder.perf.models import (
    PerformanceAlertTesting,
    PerformanceAlertSummaryTesting,
    PerformanceDatum,
    PerformanceDatumReplicate,
)

from treeherder.perf.methods import (
    CramerVonMisesDetector,
    KolmogorovSmirnovDetector,
    LeveneDetector,
    StudentTMagDetector,
    WelchDetector,
    MannWhitneyUDetector,
)

from treeherder.perf.models import PerformanceTelemetrySignature

from treeherder.perfalert.perfalert import RevisionDatum
from treeherder.services import taskcluster

from treeherder.perf.alerts import send_alert_emails, geomean, get_alert_properties

logger = logging.getLogger(__name__)

def define_methods():

    welch = WelchDetector.WelchDetector(
            min_back_window=12,
            max_back_window=24,
            fore_window=12,
            alert_threshold=2.0,
            alpha_threshold=0.005,
            mag_check=False,
            above_threshold_is_anomaly=False,
        )
    studenttmag = StudentTMagDetector.StudentTMagDetector(
            min_back_window=12,
            max_back_window=36,
            fore_window=12,
            alert_threshold=2.0,
            alpha_threshold=5,
            mag_check=True,
            above_threshold_is_anomaly=True,
        )
    levene = LeveneDetector.LeveneDetector(
            min_back_window=12,
            max_back_window=36,
            fore_window=6,
            alert_threshold=2.0,
            alpha_threshold=0.005,
            mag_check=False,
            above_threshold_is_anomaly=False,
        )
    ks = KolmogorovSmirnovDetector.KolmogorovSmirnovDetector(
            min_back_window=12,
            max_back_window=24,
            fore_window=3,
            alert_threshold=2.0,
            alpha_threshold=0.005,
            mag_check=False,
            above_threshold_is_anomaly=False,
        )
    cvm = CramerVonMisesDetector.CramerVonMisesDetector(
            min_back_window=12,
            max_back_window=24,
            fore_window=6,
            alert_threshold=2.0,
            alpha_threshold=0.005,
            mag_check=False,
            above_threshold_is_anomaly=False,
        )
    mwu = MannWhitneyUDetector.MannWhitneyUDetector(
            min_back_window=12,
            max_back_window=48,
            fore_window=3,
            alert_threshold=2.0,
            alpha_threshold=0.005,
            mag_check=False,
            above_threshold_is_anomaly=False,
        )
    methods = {
        "welch": welch,
        "levene": levene,
        "ks": ks,
        "mwu": mwu,
        "studenttmag": studenttmag,
        # "cvm": cvm
    }
    return methods

def create_alerting(signature, method, analyzed_series):

    telemetry_sig, _ = PerformanceTelemetrySignature.objects.get_or_create(
        channel=PerformanceTelemetrySignature.NIGHTLY,
        probe='test_probe',
        probe_type=PerformanceTelemetrySignature.GLEAN,
        platform=signature.platform,
        application=signature.application
    )
    for prev, cur in zip(analyzed_series, analyzed_series[1:]):
        if cur.change_detected:
            prev_value = cur.historical_stats["avg"]
            new_value = cur.forward_stats["avg"]

            alert_properties = get_alert_properties(
                prev_value, new_value, signature.lower_is_better
            )
            noise_profile = "N/A"
            try:
                # Gather all data up to the current data point that
                # shows the regression and obtain a noise profile on it.
                # This helps us to ignore this alert and others in the
                # calculation that could influence the profile.
                noise_data = []
                for point in analyzed_series:
                    if point == cur:
                        break
                    noise_data.append(geomean(point.values))
                noise_profile, _ = moz_measure_noise.deviance(noise_data)

                if not isinstance(noise_profile, str):
                    raise Exception(
                        f"Expecting a string as a noise profile, got: {type(noise_profile)}"
                    )
            except Exception:
                pass
                # Fail without breaking the alert computation
                # newrelic.agent.notice_error()
                # logger.error("Failed to obtain a noise profile.")

            summary, _ = PerformanceAlertSummaryTesting.objects.get_or_create(
                repository=signature.repository,
                framework=signature.framework,
                push_id=cur.push_id,
                prev_push_id=prev.push_id,
                sheriffed=not signature.monitor,
                defaults={
                    "manually_created": False,
                    "created": datetime.utcfromtimestamp(cur.push_timestamp),
                },
            )

            # django/mysql doesn't understand "inf", so just use some
            # arbitrarily high value for that case
            confidence = cur.t
            if confidence == float("inf"):
                confidence = 1000

            alert, _ = PerformanceAlertTesting.objects.update_or_create(
                summary=summary,
                series_signature=signature,
                telemetry_series_signature=telemetry_sig,
                detection_method=method,
                sheriffed=not signature.monitor,
                defaults={
                    "noise_profile": noise_profile,
                    "is_regression": alert_properties.is_regression,
                    "amount_pct": alert_properties.pct_change,
                    "amount_abs": alert_properties.delta,
                    "prev_value": prev_value,
                    "new_value": new_value,
                    "prev_median": 0,
                    "new_median": 0,
                    "prev_p90": 0,
                    "new_p90": 0,
                    "prev_p95": 0,
                    "new_p95": 0,
                    "confidence": confidence,
                },
            )

            # Email notifications getting disabled to not bother Sheriffs while doing testing
            # if signature.alert_notify_emails:
            #     send_alert_emails(signature.alert_notify_emails.split(), alert, summary)


def detect_methods_changes(signature, data, methods):
    methods_results = dict()
    for method_name, method_impl in methods.items():
        analyzed_series = method_impl.detect_changes(data, signature)
        methods_results[method_name] = analyzed_series
    return methods_results

def generate_test_alerts_in_series(signature):
    # get series data starting from either:
    # (1) the last alert, if there is one
    # (2) the alerts max age
    # (use whichever is newer)
    max_alert_age = alert_after_ts = datetime.now() - settings.PERFHERDER_ALERTS_MAX_AGE
    series = PerformanceDatum.objects.filter(signature=signature, push_timestamp__gte=max_alert_age)
    latest_alert_timestamp = (
        PerformanceAlertTesting.objects.filter(series_signature=signature)
        .select_related("summary__push__time")
        .order_by("-summary__push__time")
        .values_list("summary__push__time", flat=True)[:1]
    )
    if latest_alert_timestamp:
        latest_ts = latest_alert_timestamp[0]
        series = series.filter(push_timestamp__gt=latest_ts)
        if latest_ts > alert_after_ts:
            alert_after_ts = latest_ts

    datum_with_replicates = (
        PerformanceDatum.objects.filter(
            signature=signature,
            repository=signature.repository,
            push_timestamp__gte=alert_after_ts,
        )
        .annotate(
            has_replicate=Exists(
                PerformanceDatumReplicate.objects.filter(performance_datum_id=OuterRef("pk"))
            )
        )
        .filter(has_replicate=True)
    )
    replicates = PerformanceDatumReplicate.objects.filter(
        performance_datum_id__in=Subquery(datum_with_replicates.values("id"))
    ).values_list("performance_datum_id", "value")
    replicates_map: dict[int, list[float]] = {}
    for datum_id, value in replicates:
        replicates_map.setdefault(datum_id, []).append(value)

    revision_data = {}
    for d in series:
        if not revision_data.get(d.push_id):
            revision_data[d.push_id] = RevisionDatum(
                int(time.mktime(d.push_timestamp.timetuple())), d.push_id, [], []
            )
        revision_data[d.push_id].values.append(d.value)
        revision_data[d.push_id].replicates.extend(replicates_map.get(d.id, []))


    # data = revision_data.values()
    data = copy.deepcopy(list(revision_data.values()))
    methods = define_methods()

    methods_results = detect_methods_changes(
        signature,
        data,
        methods
    )

    with transaction.atomic():
        for method_name, analyzed_series in methods_results.items():
            create_alerting(signature, method_name, analyzed_series)