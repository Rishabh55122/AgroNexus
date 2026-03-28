/**
 * Full end-to-end test
 * Run: node test_e2e.js
 */
const API = 'http://localhost:8000'
const UI  = 'http://localhost:8000/ui'

async function check(name, fn) {
  try {
    await fn()
    console.log(`✅ ${name}`)
  } catch (e) {
    console.log(`❌ ${name}: ${e.message}`)
    process.exitCode = 1
  }
}

async function get(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

;(async () => {
  console.log('='.repeat(52))
  console.log('  End-to-End Integration Test — Precision Agri')
  console.log('='.repeat(52))

  // ── Backend ──────────────────────────────────────────
  await check('Backend alive',
    async () => {
      const d = await get(`${API}/`)
      if (!d.name) throw new Error('no name field')
    })

  await check('GET /tasks returns 3 tasks',
    async () => {
      const d = await get(`${API}/tasks`)
      const count = d.tasks?.length ?? d.length
      if (count !== 3) throw new Error(`got ${count}`)
    })

  await check('GET /state returns task_id',
    async () => {
      const d = await get(`${API}/state?task_id=task_1_easy`)
      if (!d.task_id) throw new Error('no task_id')
    })

  await check('Task 1 reset → step → done → grade',
    async () => {
      await post(`${API}/reset`, { task_id: 'task_1_easy' })
      let done = false, steps = 0
      while (!done && steps < 35) {
        const d = await post(`${API}/step`, { task_id: 'task_1_easy', action_type: 'wait', days: 1 })
        done = d.done
        steps++
      }
      if (!done) throw new Error('episode never finished')
      const g = await post(`${API}/grader`, { task_id: 'task_1_easy' })
      const s = g.grader_result.final_score
      if (s < 0 || s > 1) throw new Error(`score out of bounds: ${s}`)
      console.log(`   Task 1 score: ${s.toFixed(3)}`)
    })

  for (const tid of ['task_1_easy', 'task_2_medium', 'task_3_hard']) {
    await check(`/simulate ${tid}`,
      async () => {
        const d = await post(`${API}/simulate`, { task_id: tid, policy: 'greedy', seed: 42 })
        const s = d.grader_result.final_score
        if (s < 0 || s > 1) throw new Error(`score out of bounds: ${s}`)
        console.log(`   ${tid} score: ${s.toFixed(3)}`)
      })
  }

  await check('POST /baseline all tasks',
    async () => {
      const d = await post(`${API}/baseline`, {})
      for (const [tid, res] of Object.entries(d.baseline_scores)) {
        if (res.final_score < 0 || res.final_score > 1)
          throw new Error(`${tid} score out of bounds`)
      }
    })

  // ── Frontend ─────────────────────────────────────────
  await check('Frontend alive at port 3000',
    async () => {
      const r = await fetch(UI)
      if (!r.ok) throw new Error(`${r.status}`)
    })

  await check('CORS — frontend can reach backend',
    async () => {
      const r = await fetch(`${API}/`, {
        headers: { 'Origin': 'http://localhost:3000' },
      })
      if (!r.ok) throw new Error('CORS blocked')
    })

  // ── Summary ───────────────────────────────────────────
  console.log('\n' + '='.repeat(52))
  if (process.exitCode === 1) {
    console.log('❌  Some tests FAILED — fix before submitting')
  } else {
    console.log('✅  All tests PASSED — ready to submit!')
  }
  console.log('='.repeat(52))
})()
