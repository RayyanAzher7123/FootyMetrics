import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { ScrollProgress } from './components/Atmosphere'
import { MatchPage } from './pages/MatchPage'
import { ScoutingPage } from './pages/ScoutingPage'
import { ValuationPage } from './pages/ValuationPage'

const PAGES = [
  ['valuation', 'Value'],
  ['matches', 'Matches'],
  ['scouting', 'Scouting'],
] as const

type Page = (typeof PAGES)[number][0]

export default function App() {
  const [page, setPage] = useState<Page>('valuation')
  const jumping = useRef(false)

  useEffect(() => {
    const first = document.getElementById('valuation')
    first?.classList.add('is-in')
    document.documentElement.classList.add('motion')
    const nodes = document.querySelectorAll('.module-section')
    const reveal = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) entry.target.classList.add('is-in')
        })
      },
      { threshold: 0.16 },
    )
    nodes.forEach((node) => reveal.observe(node))
    return () => reveal.disconnect()
  }, [])

  useEffect(() => {
    const nodes = PAGES.map(([id]) => document.getElementById(id)).filter(
      (node): node is HTMLElement => node != null,
    )
    const observer = new IntersectionObserver(
      (entries) => {
        if (jumping.current) return
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
        const id = visible?.target.id
        if (id === 'valuation' || id === 'matches' || id === 'scouting') setPage(id)
      },
      { rootMargin: '-28% 0px -46% 0px', threshold: [0.15, 0.4, 0.7] },
    )
    nodes.forEach((node) => observer.observe(node))
    return () => observer.disconnect()
  }, [])

  function jump(id: Page) {
    setPage(id)
    jumping.current = true
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    const url = new URL(window.location.href)
    url.hash = id
    history.replaceState(null, '', url)
    window.setTimeout(() => {
      jumping.current = false
    }, 800)
  }

  function magnet(event: PointerEvent<HTMLButtonElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - (box.left + box.width / 2)
    const y = event.clientY - (box.top + box.height / 2)
    event.currentTarget.style.transform = `translate(${x * 0.28}px, ${y * 0.4}px)`
  }

  function unmagnet(event: PointerEvent<HTMLButtonElement>) {
    event.currentTarget.style.transform = ''
  }

  function spotlight(event: PointerEvent<HTMLDivElement>) {
    const target = event.target
    if (!(target instanceof Element)) return
    const panel = target.closest('.glass-panel, .hero-slab, .fixture-board, .similar-card')
    if (!(panel instanceof HTMLElement)) return
    const box = panel.getBoundingClientRect()
    panel.style.setProperty('--sx', `${event.clientX - box.left}px`)
    panel.style.setProperty('--sy', `${event.clientY - box.top}px`)
  }

  return (
    <div className="stadium" onPointerMove={spotlight}>
      <ScrollProgress />
      <div className="flood flood-left" />
      <div className="flood flood-right" />
      <div className="orb orb-a" />
      <div className="orb orb-b" />

      <div className="stage">
        <header className="site-header sticky top-0 z-20 px-4 pt-4 pb-3 md:px-8">
          <div className="glass-panel mx-auto flex max-w-6xl items-center justify-between px-4 py-3 md:px-6">
            <div>
              <p className="font-display text-3xl font-extrabold tracking-wide text-lime md:text-4xl">
                FootyMetrics
              </p>
              <p className="text-[10px] uppercase tracking-[0.28em] text-white/50">
                Premier League intelligence
              </p>
            </div>
            <nav className="flex rounded-full bg-black/30 p-1 text-sm font-medium text-white/70">
              {PAGES.map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => jump(id)}
                  onPointerMove={magnet}
                  onPointerLeave={unmagnet}
                  className={`nav-pill rounded-full px-3 py-1.5 sm:px-4 ${
                    page === id ? 'bg-lime text-pitch-deep shadow-[0_0_24px_rgba(214,255,74,0.45)]' : 'hover:text-white'
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        <main className="mx-auto max-w-6xl px-4 py-8 md:px-8 md:py-10">
          <section id="valuation" className="module-section">
            <ValuationPage />
          </section>
          <section id="matches" className="module-section">
            <MatchPage />
          </section>
          <section id="scouting" className="module-section">
            <ScoutingPage />
          </section>
        </main>
      </div>
    </div>
  )
}
