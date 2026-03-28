import json, sys

with open('baseline_results.json') as f:
    results = json.load(f)

print('Baseline results from baseline_results.json:')
all_ok = True
for tid, res in results.items():
    if 'error' in res:
        print(f'  FAIL {tid}: ERROR - {res["error"]}')
        all_ok = False
    else:
        s = res['final_score']
        ok = 0.0 <= s <= 1.0
        mark = 'OK' if ok else 'FAIL'
        print(f'  {mark} {tid}: score={s:.4f} reward={res["episode_reward"]:.4f}')
        if not ok:
            all_ok = False

print()
if all_ok:
    print('STEP 7 PASS: baseline_results.json valid')
else:
    print('STEP 7 FAIL: Fix errors')
    sys.exit(1)
