import { createContext } from 'react'

export const AppContext = createContext({ refreshKey: 0, refresh: () => {} })
