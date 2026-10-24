import { useRef, type PointerEvent, type ReactNode } from 'react'

type TiltProps = {
  children: ReactNode
  className?: string
  strength?: number
}

/** Tilts a panel toward the pointer so it reads as a physical object. */
export function Tilt({ children, className = '', strength = 10 }: TiltProps) {
  const ref = useRef<HTMLDivElement>(null)

  function move(event: PointerEvent<HTMLDivElement>) {
    const el = ref.current
    if (!el) return
    const box = el.getBoundingClientRect()
    const x = (event.clientX - box.left) / box.width - 0.5
    const y = (event.clientY - box.top) / box.height - 0.5
    el.style.transform = `rotateY(${x * strength}deg) rotateX(${-y * strength}deg) translateZ(12px)`
  }

  function reset() {
    if (ref.current) ref.current.style.transform = ''
  }

  return (
    <div className={`tilt-wrap ${className}`}>
      <div ref={ref} className="tilt-panel" onPointerMove={move} onPointerLeave={reset}>
        {children}
      </div>
    </div>
  )
}
