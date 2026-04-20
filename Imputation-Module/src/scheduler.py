"""Long-running daemon that imputes every building nightly at 23:50 Europe/Paris.

At each tick we compute tomorrow's date (the prediction target), wipe the
previous day's reconstruction artefacts (CSVs and PNGs) from /io/, and
subprocess-call impute_cli.py once per building. A failure on one building is
logged but does not abort the rest of the batch.

Also exposes a one-shot manual trigger via ``--run-now`` (exits after the
batch) and an opt-in ``--with-plots`` flag that appends ``--plot`` to each
per-building call so PNGs land next to the CSVs.
"""

from __future__ import annotations

import argparse
import functools
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
RECONSTRUCTION_GLOBS = (
    os.path.join(IO_DIR, "reconstructed_*.csv"),
    os.path.join(IO_DIR, "reconstructed_*.png"),
)
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
    for pattern in RECONSTRUCTION_GLOBS:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                log.info("Removed stale output %s", path)
            except OSError as exc:
                log.warning("Could not remove %s: %s", path, exc)


def _tomorrow_iso() -> str:
    tz = pytz.timezone(config.TIMEZONE)
    return (datetime.now(tz) + timedelta(days=1)).date().isoformat()


def run_daily_imputation(
    with_plots: bool = False,
    overlay_prior_week: bool = False,
    overlay_actual: bool = False,
) -> list:
    target_date = _tomorrow_iso()
    log.info(
        "=== Daily imputation run for target_date=%s "
        "(with_plots=%s overlay_prior_week=%s overlay_actual=%s) ===",
        target_date, with_plots, overlay_prior_week, overlay_actual,
    )

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
        if with_plots:
            plot_path = os.path.join(
                IO_DIR, f"reconstructed_{building}_{target_date}.png"
            )
            cmd.extend(["--plot", plot_path])
        if overlay_prior_week:
            cmd.append("--overlay-prior-week")
        if overlay_actual:
            cmd.append("--overlay-actual")
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
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daemon that runs the imputer batch nightly, or one-shot trigger "
            "with --run-now. --with-plots enables PNG output per building."
        )
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help=(
            "Execute the batch once, immediately, then exit. Skips the "
            "BlockingScheduler. Exit code is non-zero if any building failed."
        ),
    )
    parser.add_argument(
        "--with-plots", action="store_true",
        help=(
            "Emit a PNG overlay plot alongside each CSV output, at "
            "/io/reconstructed_<building>_<target_date>.png. Off by default "
            "so the nightly cron run stays CSV-only."
        ),
    )
    parser.add_argument(
        "--overlay-prior-week", action="store_true",
        help=(
            "Overlay the 7 days preceding the reconstructed window on the "
            "plot (shifted +7 days) as a 'copy last week' baseline. "
            "Only meaningful with --with-plots."
        ),
    )
    parser.add_argument(
        "--overlay-actual", action="store_true",
        help=(
            "Overlay the pre-imputation measured values on the plot. "
            "Only meaningful with --with-plots."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    run_kwargs = dict(
        with_plots=args.with_plots,
        overlay_prior_week=args.overlay_prior_week,
        overlay_actual=args.overlay_actual,
    )

    if args.run_now:
        log.info(
            "Mode=run-now %s Buildings=%s",
            run_kwargs, config.BUILDINGS,
        )
        failures = run_daily_imputation(**run_kwargs)
        sys.exit(1 if failures else 0)

    scheduler = BlockingScheduler(timezone=config.TIMEZONE)
    trigger = CronTrigger(
        hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, timezone=config.TIMEZONE,
    )
    scheduler.add_job(
        functools.partial(run_daily_imputation, **run_kwargs),
        trigger=trigger,
        id="daily_imputation",
        name="Daily imputation for all buildings",
        misfire_grace_time=3600,
        coalesce=True,
    )
    log.info(
        "Mode=daemon %s Cron=%02d:%02d %s Buildings=%s Next run=%s",
        run_kwargs,
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
