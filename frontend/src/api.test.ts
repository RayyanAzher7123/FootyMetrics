import { afterEach, describe, expect, it, vi } from 'vitest'
import { searchPlayers } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('searchPlayers', () => {
  it('sends a trimmed query to the players endpoint', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => [{ player_name: 'Bukayo Saka' }],
      text: async () => '',
    }))
    vi.stubGlobal('fetch', fetchMock)

    const rows = await searchPlayers('  Saka  ')

    expect(rows[0]?.player_name).toBe('Bukayo Saka')
    const url = String(fetchMock.mock.calls[0]?.[0])
    expect(url).toContain('/api/players?')
    expect(url).toContain('q=Saka')
    expect(url).toContain('limit=50')
  })

  it('raises the response body when the request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        text: async () => 'Player not found',
        json: async () => ({}),
      })),
    )
    await expect(searchPlayers('missing')).rejects.toThrow('Player not found')
  })
})
