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
    next_run = (now - pd.Timedelta(seconds=10)).floor("15min") + pd.Timedelta(minutes=15, seconds=10)
    if next_run.timestamp() > STOP_AT:
        break
    wait = (next_run - now).total_seconds()
    if wait > 0:
        time.sleep(wait)
    for attempt in range(8):                      # retry if the newest price candle isn't ready yet
        try:
            predict.main()
            break
        except Exception:
            traceback.print_exc()
            time.sleep(5)
    sh("git add predictions.csv")
    if sh("git diff --cached --quiet") != 0:
        sh('git commit -m "log"')
        sh("git pull --rebase")
        sh("git push")

print("Session finished; the next hourly start will take over.")
