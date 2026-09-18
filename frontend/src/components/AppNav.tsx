import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: 'Trade', end: true },
  { to: '/build-test', label: 'Build & Test', end: false },
  { to: '/system', label: 'System', end: false },
];

export default function AppNav({ right }: { right?: ReactNode }) {
  return (
    <nav className="ta-navbar">
      <NavLink className="ta-navbar__brand" to="/">
        <div className="ta-navbar__brand-icon">TA</div>
        <span>TradeAgent</span>
      </NavLink>

      <div className="ta-navbar__nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => `ta-navbar__link${isActive ? ' ta-navbar__link--active' : ''}`}
          >
            {item.label}
          </NavLink>
        ))}
      </div>

      <div className="ta-navbar__spacer" />
      {right ? <div className="ta-navbar__status">{right}</div> : null}
    </nav>
  );
}
