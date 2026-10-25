import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ValueCompare } from './ValueCompare'

describe('ValueCompare', () => {
  it('states when the estimate sits under the quote', () => {
    const html = renderToStaticMarkup(
      <ValueCompare estimate={75_000_000} actual={200_000_000} />,
    )
    expect(html).toContain('€125.0M under the Transfermarkt quote')
    expect(html).toContain('€200.0M')
    expect(html).toContain('>Market<')
  })

  it('hides the market bar when no squad quote was matched', () => {
    const html = renderToStaticMarkup(
      <ValueCompare estimate={10_000_000} actual={null} />,
    )
    expect(html).toContain('no matched squad quote')
    expect(html).not.toContain('>Market<')
    expect(html).not.toContain('Transfermarkt quote')
  })

  it('calls a small gap in line with the quote', () => {
    const html = renderToStaticMarkup(
      <ValueCompare estimate={100_000_000} actual={105_000_000} />,
    )
    expect(html).toContain('In line with the Transfermarkt quote')
  })
})
