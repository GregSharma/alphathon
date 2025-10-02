"""
CSP TQDM Progress Bar Wrapper

Provides a progress bar for CSP graph execution in historical mode (realtime=False).
Shows progress from starttime to endtime with ETA.

Usage:
    from graphing.csp_tqdm import run_with_progress

    run_with_progress(
        my_graph,
        starttime=start_ts,
        endtime=end_ts,
        realtime=False,
        update_interval=timedelta(seconds=1)  # optional, defaults to 1s
    )
"""

import sys
from datetime import datetime, timedelta
from typing import Callable, Optional

import csp
from csp import ts
from tqdm import tqdm


@csp.node
def _progress_tracker(
    trigger: ts[object],
    pbar: object,
    start_time: datetime,
    end_time: datetime,
    tz_offset_seconds: float,
):
    """
    Internal node that updates the tqdm progress bar.

    This node triggers periodically and updates the progress bar based on
    the current engine time relative to start and end times.

    Args:
        tz_offset_seconds: Offset in seconds to convert from UTC back to original timezone
    """
    with csp.start():
        # Initialize the progress bar total
        # Handle timezone-aware datetimes by converting to naive
        st = (
            start_time.replace(tzinfo=None)
            if hasattr(start_time, "tzinfo") and start_time.tzinfo
            else start_time
        )
        et = (
            end_time.replace(tzinfo=None)
            if hasattr(end_time, "tzinfo") and end_time.tzinfo
            else end_time
        )
        total_seconds = (et - st).total_seconds()
        pbar.total = total_seconds
        pbar.refresh()

    with csp.stop():
        # Ensure progress bar reaches 100% at the end
        et = (
            end_time.replace(tzinfo=None)
            if hasattr(end_time, "tzinfo") and end_time.tzinfo
            else end_time
        )
        pbar.n = pbar.total
        pbar.set_postfix_str(f"Completed: {et.strftime('%H:%M:%S')}")
        pbar.refresh()
        pbar.close()

    if csp.ticked(trigger):
        # Get current engine time (always in UTC)
        current_time = csp.now()

        # Handle timezone-aware datetimes by converting to naive
        st = (
            start_time.replace(tzinfo=None)
            if hasattr(start_time, "tzinfo") and start_time.tzinfo
            else start_time
        )
        et = (
            end_time.replace(tzinfo=None)
            if hasattr(end_time, "tzinfo") and end_time.tzinfo
            else end_time
        )
        ct = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )

        # Apply timezone offset to display time in original timezone
        ct_display = ct + timedelta(seconds=tz_offset_seconds)

        elapsed_seconds = (ct - st).total_seconds()

        # Update progress bar with current time in postfix
        pbar.n = elapsed_seconds
        pbar.set_postfix_str(f"Current: {ct_display.strftime('%H:%M:%S')}")
        pbar.refresh()


def run_with_progress(
    g: Callable,
    *args,
    starttime: datetime,
    endtime: datetime,
    realtime: bool = False,
    update_interval: Optional[timedelta] = None,
    pbar_desc: str = "CSP Graph Progress",
    pbar_unit: str = "s",
    output_numpy: bool = False,
    queue_wait_time: Optional[timedelta] = None,
    **kwargs,
):
    """
    Run a CSP graph with a TQDM progress bar.

    This wrapper injects a progress tracking node into your graph that updates
    a progress bar based on the engine's current time. Only works in historical
    mode (realtime=False).

    Args:
        g: The CSP graph function to run
        *args: Positional arguments to pass to the graph
        starttime: Start time for the graph execution
        endtime: End time for the graph execution
        realtime: If True, runs in realtime mode (progress bar disabled)
        update_interval: How often to update the progress bar (default: 1 second)
        pbar_desc: Description for the progress bar
        pbar_unit: Unit to display in the progress bar
        output_numpy: If True, will return each output as numpy arrays (csp.run parameter)
        queue_wait_time: Queue wait time for the engine (csp.run parameter)
        **kwargs: Additional keyword arguments to pass to the graph function

    Returns:
        The output from csp.run()
    """
    if realtime:
        # Progress bar doesn't make sense in realtime mode
        return csp.run(
            g,
            *args,
            starttime=starttime,
            endtime=endtime,
            realtime=realtime,
            output_numpy=output_numpy,
            queue_wait_time=queue_wait_time,
            **kwargs,
        )

    # Default update interval of 1 second
    if update_interval is None:
        update_interval = timedelta(seconds=1)

    # Create a wrapper graph that includes progress tracking
    # Create progress bar OUTSIDE the graph so it's available before execution starts
    # Handle timezone-aware datetimes
    st_naive = (
        starttime.replace(tzinfo=None)
        if hasattr(starttime, "tzinfo") and starttime.tzinfo
        else starttime
    )
    et_naive = (
        endtime.replace(tzinfo=None) if hasattr(endtime, "tzinfo") and endtime.tzinfo else endtime
    )

    # Calculate timezone offset (if any)
    # If starttime is tz-aware, calculate offset from UTC
    tz_offset_seconds = 0.0
    if hasattr(starttime, "tzinfo") and starttime.tzinfo:
        # Get the offset from UTC for this timezone
        utc_offset = starttime.utcoffset()
        if utc_offset:
            tz_offset_seconds = utc_offset.total_seconds()

    total_seconds = (et_naive - st_naive).total_seconds()
    pbar = tqdm(
        total=total_seconds,
        desc=pbar_desc,
        unit=pbar_unit,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix} [{elapsed}<{remaining}]",
        file=sys.stdout,
        mininterval=0.1,  # Update at most every 100ms
        miniters=0,  # Update on every iteration
    )
    pbar.set_postfix_str(
        f"Start: {st_naive.strftime('%H:%M:%S')} → End: {et_naive.strftime('%H:%M:%S')}"
    )

    # Start progress bar before run
    pbar.set_postfix_str("Starting...")
    pbar.refresh()

    # Create a wrapper graph that injects the progress tracker
    @csp.graph
    def _wrapped_graph():
        # Call the user's graph (will use csp.add_graph_output internally)
        g(*args, **kwargs)

        # Inject progress tracker with a timer trigger
        trigger = csp.timer(update_interval, True)
        _progress_tracker(trigger, pbar, starttime, endtime, tz_offset_seconds)

    try:
        result = csp.run(
            _wrapped_graph,
            starttime=starttime,
            endtime=endtime,
            realtime=realtime,
            output_numpy=output_numpy,
            queue_wait_time=queue_wait_time,
        )

        # pbar should already be closed by _progress_tracker's csp.stop()
        # But ensure it's closed in case something went wrong
        if not pbar.disable and pbar.n < pbar.total:
            pbar.n = pbar.total
            pbar.set_postfix_str("Completed")
            pbar.refresh()
        pbar.close()

        return result
    except Exception as e:
        pbar.close()
        raise e
