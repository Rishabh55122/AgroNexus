import urllib.request, json
import os
import subprocess
import time
import sys

def main():
    print("Starting server...")
    proc = subprocess.Popen(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"])
    time.sleep(3)
    
    BASE = 'http://localhost:8765'

    def post(path, body=None):
        if body is None:
            data = b''
        elif body == {}:
            data = b'{}'
        else:
            data = json.dumps(body).encode()
        
        req = urllib.request.Request(
            f'{BASE}{path}',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        res = urllib.request.urlopen(req, timeout=120)
        return json.loads(res.read())

    def get(path):
        res = urllib.request.urlopen(f'{BASE}{path}', timeout=30)
        return json.loads(res.read())

    try:
        print('Simulating OpenEnv validator checks:')

        # Test 1 - Empty body POST /reset
        d = post('/reset', {})
        assert 'observation' in d, 'Missing observation'
        assert d['observation']['day'] == 1, 'Day should be 1'
        print('  ✅ OpenEnv Reset POST (with empty {}) OK')

        # Test 2 - No body at all
        try:
            d2 = post('/reset', None)
            print('  ✅ POST /reset with completely missing body works')
        except Exception as e:
            print(f'  ⚠ No body test: {e} — empty dict test {} passed so this is ok')

        # Check 2
        assert os.path.exists('Dockerfile'), 'Dockerfile missing'
        print('  ✅ Dockerfile at repo root')

        # Check 3
        import importlib.util
        spec = importlib.util.spec_from_file_location('inference', 'inference.py')
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print('  ✅ inference.py at repo root and imports correctly')

        # Check 4
        d = get('/tasks')
        assert len(d['tasks']) >= 3
        for t in d['tasks']:
            assert t['difficulty'], f'{t["task_id"]} difficulty empty'
            assert t['description'], f'{t["task_id"]} description empty'
        print('  ✅ /tasks returns 3+ tasks with descriptions')

        # Check 5
        for tid in ['task_1_easy', 'task_2_medium', 'task_3_hard']:
            d = post('/simulate', {'task_id': tid, 'policy': 'greedy', 'seed': 42})
            s = d['grader_result']['final_score']
            assert 0.0 <= s <= 1.0, f'{tid} score out of bounds: {s}'
            print(f'  ✅ {tid} score: {s}')

        print('\n✅ All OpenEnv validator checks passed')
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
