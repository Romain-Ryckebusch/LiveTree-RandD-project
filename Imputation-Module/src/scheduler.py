"""Long-running daemon that imputes every building nightly at 23:50 Europe/Paris.

At each tick we compute tomorrow's date (the prediction target), wipe the
previous day's reconstruction CSVs from /io/, and subprocess-call impute_cli.py
once per building. A failure on one building is logged but does not abort the
rest of the batch.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config


IO_DIR = "/io"
RECONSTRUCTION_GLOB = os.path.join(IO_DIR, "reconstructed_*.csv")
CLI_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impute_cli.py")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not an integer (expected {lo}-{hi})")
    if not lo <= value <= hi:
        raise SystemExit(f"{name}={value} out of range (expected {lo}-{hi})")
    return value


SCHEDULE_HOUR = _env_int("SCHEDULE_HOUR", 23, 0, 23)
SCHEDULE_MINUTE = _env_int("SCHEDULE_MINUTE", 50, 0, 59)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("imputer-scheduler")


def _wipe_previous_outputs() -> None:
    for path in glob.glob(RECONSTRUCTION_GLOB):
        try:
            os.remove(path)
            log.info("Removed stale output %s", path)
        except OSError as exc:
            log.warning("Could not remove %s: %s", path, exc)


def _tomorrow_iso() -> str:
    tz = pytz.timezone(config.TIMEZONE)
    return (datetime.now(tz) + timedelta(days=1)).date().isoformat()


def run_daily_imputation() -> None:
    target_date = _tomorrow_iso()
    log.info("=== Daily imputation run for target_date=%s ===", target_date)

    _wipe_previous_outputs()

    successes = 0
    failures = []
    for building in config.BUILDINGS:
        output_path = os.path.join(IO_DIR, f"reconstructed_{building}_{target_date}.csv")
        cmd = [
            sys.executable, CLI_SCRIPT,
            "--source", "cassandra",
            "--target-date", target_date,
            "--building", building,
            "--output", output_path,
        ]
        log.info("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True,
            )
        except Exception as exc:
            log.exception("Subprocess launch failed for %s: %s", building, exc)
            failures.append(building)
            continue

        if result.stdout:
            log.info("[%s stdout]\n%s", building, result.stdout.rstrip())
        if result.stderr:
            log.info("[%s stderr]\n%s", building, result.stderr.rstrip())

        if result.returncode == 0:
            successes += 1
            log.info("[OK] %s -> %s", building, output_path)
        else:
            failures.append(building)
            log.error("[FAIL] %s exited with code %d", building, result.returncode)

    log.info(
        "=== Run complete: %d/%d succeeded, failures=%s ===",
        successes, len(config.BUILDINGS), failures or "none",
    )


def main() -> None:
    scheduler = BlockingScheduler(timezone=config.TIMEZONE)
    trigger = CronTrigger(
        hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, timezone=config.TIMEZONE,
    )
    scheduler.add_job(
        run_daily_imputation,
        trigger=trigger,
        id="daily_imputation",
        name="Daily imputation for all buildings",
        misfire_grace_time=3600,
        coalesce=True,
    )
    log.info(
        "Scheduler started. Cron=%02d:%02d %s Buildings=%s Next run=%s",
        SCHEDULE_HOUR, SCHEDULE_MINUTE, config.TIMEZONE,
        config.BUILDINGS,
        trigger.get_next_fire_time(None, datetime.now(pytz.timezone(config.TIMEZONE))),
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler shutting down.")


if __name__ == "__main__":
    main()
