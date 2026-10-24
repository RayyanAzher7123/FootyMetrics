export type ClubIdentity = {
  short: string
  primary: string
  secondary: string
  text: string
  crest: string
}

const CLUBS: Record<string, ClubIdentity> = {
  Arsenal: {
    short: 'ARS',
    primary: '#EF0107',
    secondary: '#063672',
    text: '#FFFFFF',
    crest: '/crests/arsenal.png',
  },
  'Aston Villa': {
    short: 'AVL',
    primary: '#670E36',
    secondary: '#95BFE5',
    text: '#FFFFFF',
    crest: '/crests/villa.png',
  },
  Bournemouth: {
    short: 'BOU',
    primary: '#DA291C',
    secondary: '#000000',
    text: '#FFFFFF',
    crest: '/crests/bournemouth.png',
  },
  Brentford: {
    short: 'BRE',
    primary: '#E30613',
    secondary: '#FBB800',
    text: '#FFFFFF',
    crest: '/crests/brentford.png',
  },
  Brighton: {
    short: 'BHA',
    primary: '#0057B8',
    secondary: '#FFCD00',
    text: '#FFFFFF',
    crest: '/crests/brighton.png',
  },
  Burnley: {
    short: 'BUR',
    primary: '#6C1D45',
    secondary: '#99D6EA',
    text: '#FFFFFF',
    crest: '/crests/burnley.png',
  },
  Chelsea: {
    short: 'CHE',
    primary: '#034694',
    secondary: '#DBA111',
    text: '#FFFFFF',
    crest: '/crests/chelsea.png',
  },
  'Crystal Palace': {
    short: 'CRY',
    primary: '#1B458F',
    secondary: '#C4122E',
    text: '#FFFFFF',
    crest: '/crests/palace.png',
  },
  Everton: {
    short: 'EVE',
    primary: '#003399',
    secondary: '#FFFFFF',
    text: '#FFFFFF',
    crest: '/crests/everton.png',
  },
  Fulham: {
    short: 'FUL',
    primary: '#000000',
    secondary: '#FFFFFF',
    text: '#FFFFFF',
    crest: '/crests/fulham.png',
  },
  Leeds: {
    short: 'LEE',
    primary: '#1D428A',
    secondary: '#FFCD00',
    text: '#FFFFFF',
    crest: '/crests/leeds.png',
  },
  Liverpool: {
    short: 'LIV',
    primary: '#C8102E',
    secondary: '#00B2A9',
    text: '#FFFFFF',
    crest: '/crests/liverpool.png',
  },
  'Manchester City': {
    short: 'MCI',
    primary: '#6CABDD',
    secondary: '#1C2C5B',
    text: '#0B1B33',
    crest: '/crests/mancity.png',
  },
  'Manchester United': {
    short: 'MUN',
    primary: '#DA291C',
    secondary: '#FBE122',
    text: '#FFFFFF',
    crest: '/crests/manutd.png',
  },
  Newcastle: {
    short: 'NEW',
    primary: '#241F20',
    secondary: '#FFFFFF',
    text: '#FFFFFF',
    crest: '/crests/newcastle.png',
  },
  'Nottingham Forest': {
    short: 'NFO',
    primary: '#DD0000',
    secondary: '#FFFFFF',
    text: '#FFFFFF',
    crest: '/crests/forest.png',
  },
  Sunderland: {
    short: 'SUN',
    primary: '#EB172B',
    secondary: '#211747',
    text: '#FFFFFF',
    crest: '/crests/sunderland.png',
  },
  Tottenham: {
    short: 'TOT',
    primary: '#132257',
    secondary: '#FFFFFF',
    text: '#FFFFFF',
    crest: '/crests/tottenham.png',
  },
  'West Ham': {
    short: 'WHU',
    primary: '#7A263A',
    secondary: '#1BB1E7',
    text: '#FFFFFF',
    crest: '/crests/westham.png',
  },
  Wolves: {
    short: 'WOL',
    primary: '#FDB913',
    secondary: '#231F20',
    text: '#231F20',
    crest: '/crests/wolves.png',
  },
}

export function clubIdentity(name: string): ClubIdentity {
  return (
    CLUBS[name] ?? {
      short: name.slice(0, 3).toUpperCase(),
      primary: '#0f6a4b',
      secondary: '#0a4532',
      text: '#f3f6f2',
      crest: '',
    }
  )
}
