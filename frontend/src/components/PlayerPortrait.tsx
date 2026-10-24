import { useEffect, useState } from 'react'

export function PlayerPortrait({
  name,
  photoUrl,
  size = 'lg',
}: {
  name: string
  photoUrl?: string | null
  size?: 'sm' | 'lg'
}) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [photoUrl])
  const box = size === 'lg' ? 'h-40 w-32' : 'h-10 w-10'
  const initials = name
    .split(' ')
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase()

  if (!photoUrl || failed) {
    return (
      <div className={`${box} grid place-items-center rounded-2xl bg-white/10 font-display text-2xl text-lime`}>
        {initials}
      </div>
    )
  }

  return (
    <img
      src={photoUrl}
      alt=""
      onError={() => setFailed(true)}
      className={`${box} rounded-2xl object-cover object-top shadow-[0_16px_40px_rgba(0,0,0,0.45)]`}
    />
  )
}

export function CountUp({ value, format }: { value: number; format: (n: number) => string }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    const start = performance.now()
    const from = 0
    const duration = 700
    let frame = 0
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - (1 - t) ** 3
      setShown(from + (value - from) * eased)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [value])
  return <>{format(Math.round(shown))}</>
}
