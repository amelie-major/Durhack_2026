/** Small storm-scale key: colour + icon + word, so it never relies on colour alone */
import { CloudLightning, CloudSun, Sun } from 'lucide-react'

const levels = [
  { name: 'Calm', Icon: Sun, cls: 'text-calm' },
  { name: 'Unsettled', Icon: CloudSun, cls: 'text-unsettled' },
  { name: 'Storm', Icon: CloudLightning, cls: 'text-storm' },
]

export default function StormLegend() {
  return (
    <ul aria-label="Storm scale" className="glass flex items-center gap-4 px-4 py-2 text-xs">
      {levels.map(({ name, Icon, cls }) => (
        <li key={name} className={`flex items-center gap-1.5 ${cls}`}>
          <Icon className="size-4" aria-hidden />
          <span className="font-medium">{name}</span>
        </li>
      ))}
      <li className="text-ink-muted flex items-center gap-1.5">
        <span className="size-3 rounded-sm bg-nodata" aria-hidden />
        No forecast
      </li>
    </ul>
  )
}
