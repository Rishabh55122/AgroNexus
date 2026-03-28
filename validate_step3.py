import sys
sys.path.insert(0, '.')
from env.environment import AgriculturalEnvironment, Action

env = AgriculturalEnvironment('task_1_easy')

print('Checking OpenEnv spec compliance:')

# reset()
assert hasattr(env, 'reset'), 'reset() missing'
obs = env.reset()
assert obs is not None, 'reset() returned None'
print('  OK reset() exists and returns observation')

# step() returns 4-tuple
assert hasattr(env, 'step'), 'step() missing'
action = Action(action_type='wait', days=1)
result = env.step(action)
assert len(result) == 4, f'step() must return 4-tuple, got {len(result)}'
obs2, reward, done, info = result
print('  OK step() returns (observation, reward, done, info)')

# state()
assert hasattr(env, 'state'), 'state() missing'
state = env.state()
assert isinstance(state, dict), 'state() must return dict'
print('  OK state() returns dict')

# Observation fields
obs_dict = obs.model_dump()
required_obs = ['day', 'season_phase', 'budget_left', 'market_prices', 'weather', 'zones']
for field in required_obs:
    assert field in obs_dict, f'Observation missing field: {field}'
    print(f'  OK Observation.{field} present')

# Reward fields
reward_dict = reward.model_dump()
required_reward = ['step_reward', 'episode_reward', 'done']
for field in required_reward:
    assert field in reward_dict, f'Reward missing field: {field}'
    print(f'  OK Reward.{field} present')

# Action fields
action_dict = action.model_dump()
assert 'action_type' in action_dict, 'Action missing action_type'
print('  OK Action.action_type present')

print('\nSTEP 3 PASS: Full OpenEnv spec compliance passed')
