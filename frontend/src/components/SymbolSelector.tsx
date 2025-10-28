import { useEffect, useMemo, useState } from 'react';
import { getSymbols, type SymbolsResponse } from '../services/api';

interface SymbolSelectorProps {
  onSymbolChange: (symbol: string) => void;
  value: string;
}

export default function SymbolSelector({ onSymbolChange, value }: SymbolSelectorProps) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [query, setQuery] = useState('');

  useEffect(() => {
    getSymbols().then((response: SymbolsResponse) => {
      const availableSymbols = (response.symbols || []).slice().sort((a, b) => a.localeCompare(b));
      setSymbols(availableSymbols);

      // If the current value is not in the list, update it to the default
      if (availableSymbols.length > 0 && !availableSymbols.includes(value)) {
        const nextSymbol = response.default ?? availableSymbols[0];
        if (nextSymbol) {
          onSymbolChange(nextSymbol);
        }
      }
    });
  }, [onSymbolChange, value]);

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onSymbolChange(event.target.value);
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return symbols;
    return symbols.filter(s => s.toLowerCase().includes(q));
  }, [symbols, query]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <input
        id="symbolSearch"
        type="text"
        placeholder="Search symbol..."
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <select title="Symbol" value={value} onChange={handleChange}>
        {filtered.map(symbol => (
          <option key={symbol} value={symbol}>
            {symbol}
          </option>
        ))}
      </select>
    </div>
  );
}
