"""Run swell + sea-state sims on Modal (the laptop's load average is >100 with ~30 lanes).
Usage (local): python3 modal_sim.py jobs.json   -- each job: name, depth npz path, params. Results -> ../cache/<name>.npz"""
import modal, io, json, sys, os
import numpy as np
app = modal.App("voxsim-study-sea-sim")
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==1.26.4", "scipy==1.14.0")
         .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sea"), "/root/sea"))

@app.function(image=image, cpu=2.0, memory=8192, timeout=3600)
def run(blob: bytes, params: dict) -> bytes:
    import sys; sys.path.insert(0, "/root")
    import numpy as np, io, time
    from sea.waves import swell, eikonal
    from sea.sim import SeaState
    t0 = time.time()
    d = dict(np.load(io.BytesIO(blob)))
    depth, z = d['depth'], d['z']
    wp = params.get('waves', {}); sp = params.get('sim', {})
    F = swell(depth, **wp)
    te, kx, ky = eikonal(depth, T=wp.get('T', 8.3), dir_from=wp.get('dir_from', 281.0))
    F['t'] = te; F['kx'] = kx; F['ky'] = ky
    S = SeaState(depth, z, F, wp.get('T', 8.3), **sp).run()
    out = dict(H=F['H'], Q=F['Q'], t=te, kx=kx, ky=ky, land=F['land'],
               W=S.W, F=S.F, A=S.A, q=S.q, Hinst=S.Hinst, brk=S.brk, kelp=S.kelp, kelp_rho=S.kelp_rho,
               rock=S.rock, rough=S.rough, u=S.u, v=S.v,
               px=S.parts['x'], py=S.parts['y'],
               t_now=np.array(S.t), secs=np.array(time.time() - t0))
    def cv(k, v):
        v = np.asarray(v)
        if v.dtype.kind == 'f' and k not in ('px', 'py', 't', 'secs', 't_now'): return v.astype(np.float16)
        return v.astype(np.float32) if v.dtype.kind == 'f' else v
    b = io.BytesIO(); np.savez_compressed(b, **{k: cv(k, v) for k, v in out.items()}); return b.getvalue()

@app.local_entrypoint()
def main(jobs: str):
    J = json.load(open(jobs))
    args = []
    for j in J:
        args.append((open(j['input'], 'rb').read(), j['params']))
    for j, res in zip(J, run.starmap(args)):
        open(j['output'], 'wb').write(res)
        print('done', j['output'], len(res))
