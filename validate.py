import json
import yaml

try:
    json.load(open('tasks/task_easy.json'))
    print('task_easy.json SUCCESS')
except Exception as e:
    print(f"task_easy.json FAILED: {e}")

try:
    json.load(open('tasks/task_medium.json'))
    print('task_medium.json SUCCESS')
except Exception as e:
    print(f"task_medium.json FAILED: {e}")

try:
    json.load(open('tasks/task_hard.json'))
    print('task_hard.json SUCCESS')
except Exception as e:
    print(f"task_hard.json FAILED: {e}")

try:
    yaml.safe_load(open('openenv.yaml'))
    print('openenv.yaml SUCCESS')
except Exception as e:
    print(f"openenv.yaml FAILED: {e}")

from env.environment import AgriculturalEnvironment, TASK_CONFIGS

for fname, tid in [
    ('tasks/task_easy.json',   'task_1_easy'),
    ('tasks/task_medium.json', 'task_2_medium'),
    ('tasks/task_hard.json',   'task_3_hard'),
]:
    try:
        data = json.load(open(fname))
        assert data['task_id'] == tid, f"task_id mismatch in {fname}"
        assert tid in TASK_CONFIGS,    f"{tid} missing from TASK_CONFIGS"
        env = AgriculturalEnvironment(tid)
        obs = env.reset()
        assert len(obs.zones) == data['num_zones'], f"Zone count mismatch for {tid}"
        print(f"{fname} -> {tid} SUCCESS")
    except Exception as e:
        print(f"{fname} -> {tid} FAILED: {e}")
