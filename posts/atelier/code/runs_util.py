import numpy as np
def top_profile(mask):
    has = mask.any(0); return np.where(has, mask.argmax(0), -1)
def runs(profile):
    p = profile[profile >= 0]; d = np.diff(p); rl = []; c = 1
    for v in d:
        if v == 0: c += 1
        else: rl.append(c); c = 1
    rl.append(c); return rl, [int(v) for v in d if v != 0]
def steady_stats(name, pr):
    rl, j = runs(pr); a = np.array(rl[1:-1]); j = np.abs(j)
    print(f"{name:10s} runs={rl[:20]} 1px-steps={np.mean(j==1):.0%} run==prev={np.mean(a[1:]==a[:-1]):.0%} run within 1 of prev={np.mean(np.abs(a[1:]-a[:-1])<=1):.0%}")
