import { useEffect, useState } from 'react';
import { getSymbols, type SymbolsResponse } from '../services/api';

interface SymbolSelectorProps {
  onSymbolChange: (symbol: string) => void;
  value: string;
}

export default function SymbolSelector({ onSymbolChange, value }: SymbolSelectorProps) {
  const [symbols, setSymbols] = useState<string[]>([]);

  useEffect(() => {
    getSymbols().then((response: SymbolsResponse) => {
      const availableSymbols = response.symbols;
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

  return (
    <select value={value} onChange={handleChange}>
      {symbols.map(symbol => (
        <option key={symbol} value={symbol}>
          {symbol}
        </option>
      ))}
    </select>
  );
}
