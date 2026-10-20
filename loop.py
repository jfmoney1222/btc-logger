import subprocess, time, traceback
import pandas as pd
import predict

STOP_AT = time.time() + (5 * 60 + 40) * 60   # stop before GitHub's 6-hour job limit


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("command failed:", cmd, r.stdout[-300:], r.stderr[-300:])
    return r.returncode


while True:
    now = pd.Timestamp.now(tz="UTC")
    next_run = (now - pd.Timedelta(seconds=60)).floor("15min") + pd.Timedelta(minutes=16)
    if next_run.timestamp() > STOP_AT:
        break
    wait = (next_run - now).total_seconds()
    if wait > 0:
        time.sleep(wait)
    try:
        predict.main()
        sh("git add predictions.csv")
        if sh("git diff --cached --quiet") != 0:
            sh('git commit -m "log"')
            sh("git pull --rebase")
            sh("git push")
    except Exception:
        traceback.print_exc()

print("Session finished; the next hourly start will take over.")
