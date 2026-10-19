import argparse
import json
import os
import subprocess
import sys
import threading
import time
import signal
from pathlib import Path


def stream_process(name, cmd, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] $ {' '.join(map(str, cmd))}", flush=True)
    p = subprocess.Popen(
        [str(x) for x in cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def pump():
        with log_path.open("a") as f, p.stdout as stream:
            for line in stream:
                f.write(line)
                f.flush()
                print(f"[{name}] {line}", end="", flush=True)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return p, t


def gpu_tokens(visible, inherited):
    if inherited is None:
        return [str(i) for i in range(visible)]
    tokens = [x.strip() for x in inherited.split(',') if x.strip()]
    if len(tokens) != visible:
        raise RuntimeError(f'Allocation lists {len(tokens)} devices but CUDA exposes {visible}; check allocation/MIG configuration')
    return tokens


def launch_pair(repo_dir, run_dir, phase, resume=False):
    cfg = json.loads((run_dir / "config.json").read_text())
    assets = run_dir / "assets.json"
    tc = cfg["training"]

    if phase == "smoke":
        output_root = run_dir / 'smoke' / str(time.time_ns())
        max_steps = int(tc["smoke_steps"])
        shuffle = int(tc["smoke_shuffle_buffer_size"])
        save_steps = max_steps
    else:
        output_root = run_dir / "training"
        max_steps = int(tc["max_steps"])
        shuffle = int(tc["shuffle_buffer_size"])
        save_steps = int(tc["save_steps"])

    output_root.mkdir(parents=True, exist_ok=True)
    visible = int(os.environ.get("VLA_VISIBLE_GPU_COUNT", "0"))
    if visible <= 0:
        import torch
        visible = torch.cuda.device_count()

    if visible < 1:
        raise RuntimeError("No CUDA GPU visible to orchestrator")

    tokens = gpu_tokens(visible, os.environ.get('CUDA_VISIBLE_DEVICES'))
    parallel = visible >= 2 and os.environ.get('VLA_SEQUENTIAL', '0') != '1' 
    conditions = [("uniform", 0), ("mmd_selective", 1 if parallel else 0)]

    def command(condition):
        cmd = [
            sys.executable, "-u", str(repo_dir / "scripts" / "train_condition.py"),
            "--condition", condition,
            "--config", str(run_dir / "config.json"),
            "--assets", str(assets),
            "--output-root", str(output_root),
            "--max-steps", str(max_steps),
            "--batch-size", str(tc["batch_size"]),
            "--grad-accum", str(tc["grad_accumulation_steps"]),
            "--shuffle-buffer", str(shuffle),
            "--save-steps", str(save_steps),
        ]
        if phase == "smoke":
            cmd.append("--smoke")
        if resume:
            cmd.append("--resume")
        return cmd

    processes = []
    if parallel:
        print("Using two independent single-GPU processes in parallel.", flush=True)
        print("No DDP is used; Uniform and MMD remain independent experiments.", flush=True)
        for condition, gpu in conditions:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = tokens[gpu]
            log = run_dir / "logs" / f"{phase}_{condition}.log"
            p, t = stream_process(condition, command(condition), env, log)
            processes.append((condition, p, t))
        try:
            pending = list(processes)
            while pending:
                for condition, proc, thread in list(pending):
                    rc = proc.poll()
                    if rc is None: continue
                    thread.join()
                    pending.remove((condition,proc,thread))
                    if rc != 0: raise RuntimeError(f'{phase} failed for {condition}, exit {rc}; stopping the other condition')
                if pending: time.sleep(0.2)
        finally:
            for _, proc, thread in processes:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                thread.join(timeout=5)
    else:
        print("Only one GPU visible; running conditions sequentially.", flush=True)
        for condition, gpu in conditions:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = tokens[gpu]
            log = run_dir / "logs" / f"{phase}_{condition}.log"
            p, t = stream_process(condition, command(condition), env, log)
            try:
                rc = p.wait()
            finally:
                if p.poll() is None:
                    p.terminate()
                    try: p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        p.kill(); p.wait()
                t.join(timeout=5)
            if rc != 0:
                raise RuntimeError(f"{phase} failed for {condition}, exit code {rc}")

    audits = [json.loads((output_root / condition / 'completion.json').read_text())['audit'] for condition,_ in conditions]
    if audits[0]['non_llm_modules'] != audits[1]['non_llm_modules']:
        raise RuntimeError('Conditions do not use the same non-decoder adapter modules')
    # Fresh-process reload verification, sequentially on the first allocated GPU.
    base = json.loads(assets.read_text())["base_model_overlay"]
    for condition, _ in conditions:
        cdir = output_root / condition
        adapter = cdir / "final_adapter"
        stats = cdir / "dataset_statistics.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = tokens[0]
        cmd = [
            sys.executable, "-u", str(repo_dir / "scripts" / "verify_adapter.py"),
            "--base", base,
            "--adapter", str(adapter),
            "--stats", str(stats),
            "--condition", condition,
        ]
        log = run_dir / "logs" / f"{phase}_{condition}_reload.log"
        p, t = stream_process(f"{condition}-reload", cmd, env, log)
        try:
            rc = p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
            t.join(timeout=5)
        if rc != 0:
            raise RuntimeError(
                f"Fresh adapter reload failed for {condition}, exit code {rc}"
            )

    (run_dir / f'{phase.upper()}_COMPLETE.json').write_text(json.dumps({
        'phase':phase, 'output_root':str(output_root), 'status':'PASS',
    },indent=2)+'\n')
    print(f"{phase.upper()} PHASE: PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--phase", choices=["smoke", "train"], required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Orchestrator terminated')))
    launch_pair(args.repo_dir.resolve(), args.run_dir.resolve(), args.phase, args.resume)


if __name__ == "__main__":
    main()
