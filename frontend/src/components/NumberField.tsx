import { useEffect, useState, type ReactNode } from 'react'

type NumberFieldProps = {
  label: string
  value: number
  min: number
  max: number
  onChange: (value: number) => void
}

/** Text-like numeric field that allows clearing without forcing 0 → "020". */
export function NumberField({ label, value, min, max, onChange }: NumberFieldProps) {
  const [text, setText] = useState(String(value))

  useEffect(() => {
    setText(String(value))
  }, [value])

  function commit(raw: string) {
    if (raw.trim() === '') {
      setText(String(min))
      onChange(min)
      return
    }
    const n = Number(raw)
    if (!Number.isFinite(n)) {
      setText(String(value))
      return
    }
    const clamped = Math.min(max, Math.max(min, Math.trunc(n)))
    setText(String(clamped))
    onChange(clamped)
  }

  return (
    <label className="block text-sm">
      <span className="mb-1.5 block font-medium text-ink/80">{label}</span>
      <input
        type="text"
        inputMode="numeric"
        pattern="[0-9]*"
        value={text}
        onChange={(e) => {
          const raw = e.target.value
          if (raw === '') {
            setText('')
            return
          }
          if (!/^\d+$/.test(raw)) return
          // Drop leading zeros while typing (020 → 20), keep a single 0 if that's the value
          const normalized = raw.replace(/^0+(?=\d)/, '')
          setText(normalized)
          const n = Number(normalized)
          if (Number.isFinite(n)) {
            onChange(Math.min(max, Math.max(min, Math.trunc(n))))
          }
        }}
        onBlur={() => commit(text)}
        className="field-input"
      />
    </label>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1.5 block font-medium text-ink/80">{label}</span>
      {children}
    </label>
  )
}
