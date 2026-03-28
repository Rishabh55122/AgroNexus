"""
API integration test — runs against live FastAPI server.
Start server first: uvicorn api.main:app --port 8000
Then run: python test_api.py
"""
import requests
import json

BASE = "http://localhost:8000"

def assert_key(d, key):
    assert key in d, f"Missing key '{key}' in response: {d}"

def test(name, response, check_fn=None):
    assert response.status_code == 200, \
        f"{name} failed: {response.status_code} {response.text}"
    data = response.json()
    if check_fn:
        check_fn(data)
    print(f"SUCCESS: {name}")
    return data


print("=" * 50)
print("API Integration Tests")
print("=" * 50)

# Test 1 — Root health check
test("GET /",
    requests.get(f"{BASE}/"),
    lambda d: (
        assert_key(d, "name"),
        assert_key(d, "status"),
    )
)

# Test 2 — GET /state
test("GET /state",
    requests.get(f"{BASE}/state?task_id=task_1_easy"),
    lambda d: assert_key(d, "task_id")
)

# Test 3 — GET /tasks
data = test("GET /tasks",
    requests.get(f"{BASE}/tasks"),
    lambda d: (
        assert_key(d, "tasks"),
    )
)
assert len(data["tasks"]) == 3, "Should return 3 tasks"
print(f"   Tasks returned: {[t['task_id'] for t in data['tasks']]}")

# Test 4 — POST /reset
data = test("POST /reset",
    requests.post(f"{BASE}/reset",
        json={"task_id": "task_1_easy"}),
    lambda d: assert_key(d, "observation")
)
obs = data["observation"]
assert obs["day"] == 1,         "Day should be 1 after reset"
assert len(obs["zones"]) == 2,  "Task 1 should have 2 zones"
print(f"   Day: {obs['day']}, Zones: {len(obs['zones'])}")

# Test 5 — POST /step irrigate
data = test("POST /step irrigate",
    requests.post(f"{BASE}/step", json={
        "task_id":     "task_1_easy",
        "action_type": "irrigate",
        "zone":        0,
        "amount":      10.0,
    }),
    lambda d: assert_key(d, "reward")
)
assert data["observation"]["day"] == 2, "Day should advance to 2"
print(f"   Step reward: {data['reward']['step_reward']}")

# Test 6 — POST /step wait
test("POST /step wait",
    requests.post(f"{BASE}/step", json={
        "task_id":     "task_1_easy",
        "action_type": "wait",
        "days":        1,
    })
)

# Test 7 — POST /simulate greedy
data = test("POST /simulate greedy",
    requests.post(f"{BASE}/simulate", json={
        "task_id": "task_1_easy",
        "policy":  "greedy",
        "seed":    42,
    }),
    lambda d: assert_key(d, "grader_result")
)
score = data["grader_result"]["final_score"]
assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
print(f"   Greedy score task_1: {score}")

# Test 8 — POST /simulate random
data = test("POST /simulate random",
    requests.post(f"{BASE}/simulate", json={
        "task_id": "task_2_medium",
        "policy":  "random",
        "seed":    42,
    }),
    lambda d: assert_key(d, "grader_result")
)
print(f"   Random score task_2: {data['grader_result']['final_score']}")

# Test 9 — POST /baseline all tasks
data = test("POST /baseline all tasks",
    requests.post(f"{BASE}/baseline", json={}),
    lambda d: assert_key(d, "baseline_scores")
)
scores = data["baseline_scores"]
for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
    s = scores[tid]["final_score"]
    assert 0.0 <= s <= 1.0, f"{tid} score out of bounds: {s}"
    print(f"   {tid}: {s}")

# Test 10 — POST /grader after simulate
requests.post(f"{BASE}/reset", json={"task_id": "task_3_hard"})
data = test("POST /simulate task_3",
    requests.post(f"{BASE}/simulate", json={
        "task_id": "task_3_hard",
        "policy":  "greedy",
        "seed":    42,
    })
)
print(f"   Hard score: {data['grader_result']['final_score']}")

# Test 11 — Invalid task_id returns 400
resp = requests.post(f"{BASE}/reset",
    json={"task_id": "task_invalid"})
assert resp.status_code == 400, \
    f"Invalid task should return 400, got {resp.status_code}"
print("SUCCESS: Invalid task_id correctly returns 400")

# Test 12 — Step before reset returns 400
env_fresh = requests.post(f"{BASE}/reset",
    json={"task_id": "task_2_medium"})
# Force a done env by running simulate first
requests.post(f"{BASE}/simulate",
    json={"task_id": "task_2_medium", "policy": "wait"})
resp = requests.post(f"{BASE}/step", json={
    "task_id":     "task_2_medium",
    "action_type": "wait",
})
assert resp.status_code == 400, \
    f"Step after done should return 400, got {resp.status_code}"
print("SUCCESS: Step after done correctly returns 400")

print("\n" + "=" * 50)
print("SUCCESS: All API tests passed")
print("=" * 50)
