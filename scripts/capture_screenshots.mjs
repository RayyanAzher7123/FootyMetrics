/**
 * Capture the three modules for docs/screenshots.
 * Expects the API on 127.0.0.1:8000 and the Vite app on 127.0.0.1:5174.
 */
import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const APP = process.env.FOOTY_APP ?? 'http://127.0.0.1:5174/'
const PORT = 9333
const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
const OUT = join(process.cwd(), 'docs', 'screenshots')

class Cdp {
  constructor(ws) {
    this.ws = ws
    this.next = 1
    this.pending = new Map()
    ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data)
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(JSON.stringify(message.error)))
      else pending.resolve(message.result)
    })
  }

  send(method, params = {}, sessionId) {
    const id = this.next++
    const payload = { id, method, params }
    if (sessionId) payload.sessionId = sessionId
    this.ws.send(JSON.stringify(payload))
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }))
  }
}

function setNativeValue(selector, value) {
  const input = document.querySelector(selector)
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
  setter.call(input, value)
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

const chrome = spawn(
  CHROME,
  [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${join(tmpdir(), 'footymetrics-shots')}`,
    '--window-size=1440,1200',
    'about:blank',
  ],
  { stdio: 'ignore' },
)

try {
  let version
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      version = await fetch(`http://127.0.0.1:${PORT}/json/version`).then((response) => response.json())
      break
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200))
    }
  }
  if (!version) throw new Error('Chrome debugging port did not open')

  const browser = new WebSocket(version.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    browser.addEventListener('open', resolve)
    browser.addEventListener('error', reject)
  })
  const cdp = new Cdp(browser)
  const { targetId } = await cdp.send('Target.createTarget', { url: APP })
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true })
  await cdp.send('Page.enable', {}, sessionId)
  await cdp.send('Runtime.enable', {}, sessionId)
  await cdp.send(
    'Emulation.setDeviceMetricsOverride',
    { width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false },
    sessionId,
  )
  await cdp.send('Page.navigate', { url: APP }, sessionId)
  await cdp.send(
    'Runtime.evaluate',
    {
      awaitPromise: true,
      expression: `new Promise((resolve) => {
        const started = Date.now()
        const tick = () => {
          if (document.querySelector('#search') && document.body.innerText.includes('Holdout MAE')) resolve('ready')
          else if (Date.now() - started > 15000) resolve('timeout')
          else setTimeout(tick, 200)
        }
        tick()
      })`,
      returnByValue: true,
    },
    sessionId,
  )

  await cdp.send(
    'Runtime.evaluate',
    {
      awaitPromise: true,
      expression: `(() => {
        ${setNativeValue.toString()}
        setNativeValue('#search', 'Haaland')
        return new Promise((resolve) => setTimeout(() => {
          const button = [...document.querySelectorAll('button')].find((node) => node.textContent.includes('Erling Haaland'))
          button?.click()
          setTimeout(resolve, 1500)
        }, 600))
      })()`,
    },
    sessionId,
  )
  await shoot(cdp, sessionId, 'valuation', '#valuation', 1680)

  await cdp.send(
    'Runtime.evaluate',
    {
      awaitPromise: true,
      expression: `(() => {
        [...document.querySelectorAll('button')].find((node) => node.textContent.trim() === 'Matches')?.click()
        return new Promise((resolve) => setTimeout(() => {
          [...document.querySelectorAll('button')].find((node) => node.textContent.includes('Predict result'))?.click()
          setTimeout(resolve, 1500)
        }, 400))
      })()`,
    },
    sessionId,
  )
  await shoot(cdp, sessionId, 'matches', '#matches', 1400)

  await cdp.send(
    'Runtime.evaluate',
    {
      awaitPromise: true,
      expression: `(() => {
        ${setNativeValue.toString()}
        [...document.querySelectorAll('button')].find((node) => node.textContent.trim() === 'Scouting')?.click()
        return new Promise((resolve) => setTimeout(() => {
          setNativeValue('input[placeholder="Search name or club…"]', '')
          const boxes = [...document.querySelectorAll('input[placeholder="Search name or club…"]')]
          const scout = boxes[boxes.length - 1]
          const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
          setter.call(scout, 'Saka')
          scout.dispatchEvent(new Event('input', { bubbles: true }))
          setTimeout(() => {
            const root = document.querySelector('#scouting')
            const button = [...root.querySelectorAll('button')].find((node) => node.textContent.includes('Bukayo Saka'))
            button?.click()
            const started = Date.now()
            const wait = () => {
              if (root.innerText.includes('Similar players') || Date.now() - started > 8000) resolve(root.innerText.includes('Similar players'))
              else setTimeout(wait, 200)
            }
            wait()
          }, 700)
        }, 400))
      })()`,
    },
    sessionId,
  )
  await shoot(cdp, sessionId, 'scouting', '#scouting', 2100)
  browser.close()
} finally {
  chrome.kill()
}

async function shoot(cdp, sessionId, name, selector, height) {
  const box = await cdp.send(
    'Runtime.evaluate',
    {
      expression: `(() => {
        const node = document.querySelector(${JSON.stringify(selector)})
        node.scrollIntoView({ block: 'start' })
        const rect = node.getBoundingClientRect()
        return { x: rect.left + window.scrollX, y: rect.top + window.scrollY, width: rect.width, height: ${height} }
      })()`,
      returnByValue: true,
    },
    sessionId,
  )
  const clip = box.result.value
  const shot = await cdp.send(
    'Page.captureScreenshot',
    { format: 'png', captureBeyondViewport: true, clip: { ...clip, scale: 1 } },
    sessionId,
  )
  await mkdir(OUT, { recursive: true })
  const destination = join(OUT, `${name}.png`)
  await writeFile(destination, Buffer.from(shot.data, 'base64'))
  console.log(destination, clip.width, clip.height)
}
