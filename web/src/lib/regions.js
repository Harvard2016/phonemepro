// Broad regions a visitor may pick, only for countries where accent varies a lot
// by region. The profile and every stored take hold one of these short codes or
// nothing; there is no typed text, so a small town can never be recorded.
// scripts/learn_regions.py holds the same table and refuses any other value.
export const REGIONS = {
  US: { NE: 'Northeast', S: 'South', MW: 'Midwest', W: 'West' },
  GB: {
    ENG_S: 'England (South)', ENG_N: 'England (North)', SCT: 'Scotland', WLS: 'Wales', NIR: 'Northern Ireland',
  },
  AU: { E: 'Eastern states and Tasmania', SA: 'South Australia', W: 'Western Australia and the Northern Territory' },
  CA: { ATL: 'Atlantic provinces', QC: 'Quebec', ON: 'Ontario', W: 'Prairies, British Columbia and the North' },
  IE: { E: 'East (Leinster)', S: 'South (Munster)', W: 'West and north (Connacht, Ulster)' },
  IN: { N: 'North', S: 'South', E: 'East and Northeast', W: 'West and Central' },
}

// [[code, name]] for a country, or an empty list when it offers no regions.
export const regionsOf = (country) => Object.entries(REGIONS[country] ?? {})

export const isRegion = (country, code) => typeof code === 'string' && Object.hasOwn(REGIONS[country] ?? {}, code)

export const regionName = (country, code) => (isRegion(country, code) ? REGIONS[country][code] : null)
