import { useEffect, useRef } from 'react'

export function ScrollProgress() {
  const bar = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = bar.current
    if (!el) return
    const paint = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      const progress = max > 0 ? window.scrollY / max : 0
      el.style.transform = `scaleX(${progress})`
    }
    paint()
    window.addEventListener('scroll', paint, { passive: true })
    window.addEventListener('resize', paint)
    return () => {
      window.removeEventListener('scroll', paint)
      window.removeEventListener('resize', paint)
    }
  }, [])

  return <div ref={bar} className="scroll-progress" aria-hidden="true" />
}
