import { useEffect, useMemo, useState } from 'react';
import { getV2Symbols, type SymbolsResponse } from '../services/api';

interface SymbolSelectorProps {
  onSymbolChange: (symbol: string) => void;
  value: string;
}

export default function SymbolSelector({ onSymbolChange, value }: SymbolSelectorProps) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [query, setQuery] = useState('');

  const PRIORITY_SYMBOLS = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'GER30', 'US30', 'US500', 'NAS100', 'BTCUSD', 'ETHUSD'];

  useEffect(() => {
    getV2Symbols().then((response: SymbolsResponse) => {
      const available = response.symbols || [];
      const priority = available.filter(s => PRIORITY_SYMBOLS.includes(s.toUpperCase()));
      const others = available.filter(s => !PRIORITY_SYMBOLS.includes(s.toUpperCase())).sort((a, b) => a.localeCompare(b));
      
      // Sort priority based on our list order
      priority.sort((a, b) => PRIORITY_SYMBOLS.indexOf(a.toUpperCase()) - PRIORITY_SYMBOLS.indexOf(b.toUpperCase()));
      
      const sorted = [...priority, ...others];
      setSymbols(sorted);
      
      if (sorted.length > 0 && !sorted.includes(value)) {
        const next = response.default ?? sorted[0];
        if (next) onSymbolChange(next);
      }
    });
  }, [onSymbolChange, value]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return symbols.slice(0, 500); // Increased limit for better coverage
    return symbols.filter((s) => s.toLowerCase().includes(q)).slice(0, 500);
  }, [symbols, query]);

  return (
    <div className="ta-symbol-selector">
      <input
        className="ta-symbol-selector__search"
        type="text"
        placeholder="Search…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <select
        className="ta-symbol-selector__select"
        title="Symbol"
        value={value}
        onChange={(e) => onSymbolChange(e.target.value)}
      >
        {filtered.map((symbol) => (
          <option key={symbol} value={symbol}>{symbol}</option>
        ))}
      </select>
    </div>
  );
}
