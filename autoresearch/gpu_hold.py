"""Hold a GPU for the autoresearch loop between runs, so another user's queue cannot take it.
    nohup python gpu_hold.py 1 > /fast/lev/gpu_hold/gpu_hold_1.log 2>&1 &
One tick every 5 s:
  * request file /fast/lev/gpu_hold/.gpu_release_<gpu> present (gpu_wait.sh touches it) -> shrink the held memory to a
    GRACE_MB block, delete the file (the acknowledgement gpu_wait.sh waits for) and keep only that block for GRACE seconds,
    so train.py can start up while the GPU still looks busy to other users' job pollers (a 1 GB footprint was not enough:
    a job landed on GPU 2 inside that window on Oct 30); as soon as our run appears the block shrinks to the footprint;
  * GPU empty and no grace period running -> allocate all but 1 GB and sit on it;
  * a process of ours on the GPU -> idle, but keep a 1 GB footprint so the GPU never looks empty to another user's job
    poller in the seconds between our run ending and the holder re-allocating (lost GPU 2 that way on Oct 30);
  * another user's process on the GPU -> idle (never allocates on top of someone else's job).
Opt-in: HOLD_FILL_MARGIN_MB=<n> also fills the memory beside our own run (all but n MiB, HOLD_FILL_DELAY_S seconds after
the run appeared, default 180) so a job queue that ignores busy GPUs fails to start there. Use it only for runs whose
memory is known to be stable: a run that grows later would crash against the fill.
Writes its pid to /fast/lev/gpu_hold/gpu_hold_<gpu>.pid; gpu_wait.sh uses it to know whether an acknowledgement is coming.
The shared directory (HOLD_DIR) lets one holder per GPU serve launchers from every checkout and worktree."""
import os, sys, time, pathlib, subprocess

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import torch

HOLD_DIR = pathlib.Path("/fast/lev/gpu_hold")
HOLD_DIR.mkdir(parents=True, exist_ok=True)
release_file = HOLD_DIR / f".gpu_release_{gpu}"
pid_file = HOLD_DIR / f"gpu_hold_{gpu}.pid"
pid_file.write_text(f"{os.getpid()}\n")
me = os.getpid()
user = os.environ.get("USER") or "leva"
GRACE = 300
FOOTPRINT_MB = 1024          # kept at all times (never looks empty)
GRACE_MB = 24 * 1024         # kept during the hand-over to our run, until our process shows up (its big allocations come later)
TICK = 2
FILL_MARGIN_MB = int(os.environ.get("HOLD_FILL_MARGIN_MB", 0))
FILL_DELAY_S = float(os.environ.get("HOLD_FILL_DELAY_S", 180))


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def procs():
    """[(pid, owner, used_mb)] of the compute processes on the GPU, excluding this one."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout
    res = []
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if int(pid) == me:
            continue
        owner = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip() or "?"
        res.append((int(pid), owner, int(mem)))
    return res


block = None
grace_block = False
foot = None
fill = None
ours_since = None
last = None
grace_until = 0.0
while True:
    ps = procs()
    ours = [p for p in ps if p[1] == user]
    theirs = [p for p in ps if p[1] != user]
    if release_file.exists():
        if fill is not None:
            fill = None
            torch.cuda.empty_cache()
        if block is not None:
            block = None
            torch.cuda.empty_cache()
            try:
                block = torch.empty(GRACE_MB * 2 ** 20, dtype=torch.uint8, device="cuda")
                grace_block = True
            except RuntimeError:
                grace_block = False
            log("released on request (grace block kept)" if grace_block else "released on request")
        release_file.unlink(missing_ok=True)            # the acknowledgement
        grace_until = time.time() + GRACE
    if block is not None and grace_block and (ours or time.time() >= grace_until):
        block = None                                    # our run is on the GPU (or the grace ran out): only the footprint stays
        grace_block = False
        torch.cuda.empty_cache()
    if foot is None and not theirs:                     # the footprint, kept through our own runs
        try:
            foot = torch.empty(FOOTPRINT_MB * 2 ** 20, dtype=torch.uint8, device="cuda")
        except RuntimeError:
            pass
    if not ours:
        ours_since = None
        if fill is not None:
            fill = None
            torch.cuda.empty_cache()
    if block is not None:
        state = "grace" if grace_block else "holding"
    elif ours:
        state = "ours"
        ours_since = ours_since or time.time()
        if FILL_MARGIN_MB and fill is None and time.time() - ours_since >= FILL_DELAY_S:
            mb = torch.cuda.mem_get_info()[0] // 2 ** 20 - FILL_MARGIN_MB
            if mb > 1024:
                try:
                    fill = torch.empty(mb * 2 ** 20, dtype=torch.uint8, device="cuda")
                    log(f"filling {mb} MiB beside our run (margin {FILL_MARGIN_MB} MiB)")
                except RuntimeError:
                    pass
    elif theirs:
        state = "theirs"
    elif time.time() < grace_until:
        state = "grace"
    else:
        mb = max(0, torch.cuda.mem_get_info()[0] // 2 ** 20 - 1024)
        try:
            block = torch.empty(mb * 2 ** 20, dtype=torch.uint8, device="cuda")
            log(f"holding {mb} MiB of GPU {gpu}")
            state = "holding"
        except RuntimeError as e:
            log(f"allocation failed ({str(e)[:60]}), retrying")
            state = "empty"
            time.sleep(10)
    if state != last:
        log(f"GPU {gpu}: {state}" + (f" {[(p[1], p[2]) for p in ps]}" if ps and state in ("ours", "theirs") else ""))
    last = state
    time.sleep(TICK)
