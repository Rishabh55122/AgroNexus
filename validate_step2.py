import yaml, sys

with open('openenv.yaml') as f:
    config = yaml.safe_load(f)

required_keys = [
    'name', 'version', 'description',
    'environment', 'tasks', 'api',
    'observation_space', 'action_space', 'reward',
]

print('Checking openenv.yaml structure:')
all_ok = True
for key in required_keys:
    exists = key in config
    v = str(config.get(key, 'MISSING'))[:60]
    mark = 'OK' if exists else 'MISSING'
    print(f'  {mark} {key}: {v}')
    if not exists:
        all_ok = False

tasks = config.get('tasks', [])
print(f'\nTasks defined: {len(tasks)}')
for t in tasks:
    print(f'  OK {t["id"]} - difficulty: {t["difficulty"]}')

env_class  = config.get('environment', {}).get('class', '')
env_module = config.get('environment', {}).get('module', '')
print(f'\nEnvironment class:  {env_class}')
print(f'Environment module: {env_module}')

try:
    assert env_class  == 'AgriculturalEnvironment', f'Wrong class: {env_class}'
    assert env_module == 'env.environment',          f'Wrong module: {env_module}'
    assert len(tasks) >= 3, f'Need 3+ tasks, got {len(tasks)}'
    print('\nSTEP 2 PASS: openenv.yaml validation passed')
except AssertionError as e:
    print(f'\nSTEP 2 FAIL: {e}')
    sys.exit(1)
