"""
Step 5 — openenv validate (manual implementation)
Runs all 10 official OpenEnv validation checks against http://localhost:8000
"""
import requests, json, sys, time

BASE = 'http://localhost:8000'
checks = []
results = {}

def check(name, fn):
    try:
        fn()
        print(f'  PASS {name}')
        checks.append(True)
        results[name] = 'PASS'
    except Exception as e:
        print(f'  FAIL {name}: {e}')
        checks.append(False)
        results[name] = f'FAIL: {e}'

print('Running openenv validate checks:\n')

# 1. Server alive
check('Server alive (GET /)',
    lambda: requests.get(f'{BASE}/', timeout=5).raise_for_status())

# 2. GET /state
check('GET /state returns 200',
    lambda: requests.get(f'{BASE}/state?task_id=task_1_easy', timeout=5).raise_for_status())

# 3. GET /tasks - 3+ tasks with action_schema
def check_tasks():
    d = requests.get(f'{BASE}/tasks', timeout=5).json()
    assert len(d['tasks']) >= 3, f'Need 3 tasks, got {len(d["tasks"])}'
    for t in d['tasks']:
        assert 'task_id'       in t, f'Missing task_id in {t}'
        assert 'action_schema' in t, f'Missing action_schema in task {t.get("task_id")}'
check('GET /tasks returns 3+ tasks with action_schema', check_tasks)

# 4. POST /reset - valid observation
def check_reset():
    d = requests.post(f'{BASE}/reset', json={'task_id': 'task_1_easy'}, timeout=10).json()
    assert 'observation' in d, f'Missing observation. Keys: {list(d.keys())}'
    obs = d['observation']
    assert obs['day'] == 1, f'Day should be 1, got {obs["day"]}'
    assert len(obs['zones']) == 2, f'Task 1 needs 2 zones, got {len(obs["zones"])}'
check('POST /reset returns valid observation', check_reset)

# 5. POST /step - returns 4-tuple
def check_step():
    requests.post(f'{BASE}/reset', json={'task_id': 'task_1_easy'}, timeout=10)
    d = requests.post(f'{BASE}/step', json={
        'task_id': 'task_1_easy',
        'action_type': 'wait',
        'days': 1,
    }, timeout=10).json()
    assert 'observation' in d, f'Missing observation. Keys: {list(d.keys())}'
    assert 'reward'      in d, 'Missing reward'
    assert 'done'        in d, 'Missing done'
    assert 'info'        in d, 'Missing info'
    assert d['observation']['day'] == 2, f'Day should advance to 2, got {d["observation"]["day"]}'
check('POST /step returns (observation, reward, done, info)', check_step)

# 6. POST /baseline - 3 scores in [0,1]
def check_baseline():
    d = requests.post(f'{BASE}/baseline', json={}, timeout=180).json()
    scores = d['baseline_scores']
    assert len(scores) == 3, f'Need 3 scores, got {len(scores)}'
    for tid, res in scores.items():
        s = res['final_score']
        assert 0.0 <= s <= 1.0, f'{tid} score out of bounds: {s}'
        print(f'      {tid}: {s:.4f}')
check('POST /baseline returns scores in [0.0, 1.0]', check_baseline)

# 7. POST /grader after simulate
def check_grader():
    requests.post(f'{BASE}/simulate', json={
        'task_id': 'task_1_easy', 'policy': 'greedy', 'seed': 42
    }, timeout=60)
    d = requests.post(f'{BASE}/grader', json={'task_id': 'task_1_easy'}, timeout=10).json()
    assert 'grader_result' in d, 'Missing grader_result'
    s = d['grader_result']['final_score']
    assert 0.0 <= s <= 1.0, f'Score out of bounds: {s}'
    print(f'      Score: {s:.4f}')
check('POST /grader returns valid score', check_grader)

# 8. POST /simulate - returns trajectory
def check_simulate():
    d = requests.post(f'{BASE}/simulate', json={
        'task_id': 'task_1_easy', 'policy': 'greedy', 'seed': 42
    }, timeout=60).json()
    assert 'trajectory'    in d, 'Missing trajectory'
    assert 'grader_result' in d, 'Missing grader_result'
    assert len(d['trajectory']) > 0, 'Empty trajectory'
    print(f'      Steps: {len(d["trajectory"])}')
check('POST /simulate returns trajectory', check_simulate)

# 9. All 3 graders score in [0,1]
def check_all_graders():
    for tid in ['task_1_easy', 'task_2_medium', 'task_3_hard']:
        d = requests.post(f'{BASE}/simulate', json={
            'task_id': tid, 'policy': 'greedy', 'seed': 42
        }, timeout=120).json()
        s = d['grader_result']['final_score']
        assert 0.0 <= s <= 1.0, f'{tid} score out of bounds: {s}'
        print(f'      {tid}: {s:.4f}')
check('All 3 graders score in [0.0, 1.0]', check_all_graders)

# 10. Invalid action returns 422
def check_invalid():
    requests.post(f'{BASE}/reset', json={'task_id': 'task_1_easy'}, timeout=10)
    r = requests.post(f'{BASE}/step', json={
        'task_id': 'task_1_easy',
        'action_type': 'INVALID_ACTION',
    }, timeout=10)
    assert r.status_code == 422, f'Expected 422, got {r.status_code}'
check('Invalid action_type returns 422', check_invalid)

# Summary
passed = sum(checks)
total  = len(checks)
print(f'\n{"=" * 50}')
print(f'Results: {passed}/{total} checks passed')
if passed == total:
    print('STEP 5 PASS: ALL CHECKS PASSED - ready to submit')
else:
    print('STEP 5 PARTIAL: Fix failing checks before submitting')
    for name, res in results.items():
        if res.startswith('FAIL'):
            print(f'  => {name}: {res}')
print(f'{"=" * 50}')
sys.exit(0 if passed == total else 1)
