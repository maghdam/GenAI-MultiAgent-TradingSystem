import type { V2PaperPosition } from './api';

export type ChartPositionLevelKind = 'entry' | 'stop_loss' | 'take_profit';

export interface ChartPositionLevel {
  key: string;
  kind: ChartPositionLevelKind;
  price: number;
  title: string;
  direction: 'long' | 'short';
  source: 'broker' | 'paper';
}

function finitePrice(value?: number | null): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function formatQuantity(value: number): string {
  if (!Number.isFinite(value)) return '';
  return value.toFixed(4).replace(/\.?0+$/, '');
}

function positionLabel(position: V2PaperPosition, source: 'broker' | 'paper'): string {
  const side = position.direction === 'long' ? 'BUY' : 'SELL';
  const quantity = formatQuantity(position.quantity);
  if (source === 'broker' && position.broker_position_id != null) {
    return `${side} ${quantity} #${position.broker_position_id}`;
  }
  return `PAPER ${side} ${quantity} #${position.id}`;
}

export function buildChartPositionLevels(
  positions: V2PaperPosition[],
  symbol: string,
): ChartPositionLevel[] {
  const normalizedSymbol = symbol.trim().toUpperCase();
  const levels: ChartPositionLevel[] = [];

  for (const position of positions) {
    if (position.status !== 'open' || position.symbol.trim().toUpperCase() !== normalizedSymbol) {
      continue;
    }

    const brokerBacked = position.broker_position_id != null;
    const source: 'broker' | 'paper' = brokerBacked ? 'broker' : 'paper';
    const identity = brokerBacked
      ? String(position.broker_position_id)
      : String(position.id);
    const label = positionLabel(position, source);
    const protectionLabelPrefix = source === 'paper' ? 'PAPER ' : '';

    const entry = finitePrice(
      brokerBacked ? position.broker_entry_price : position.entry_price,
    );
    const stopLoss = finitePrice(
      brokerBacked ? position.broker_stop_loss : position.stop_loss,
    );
    const takeProfit = finitePrice(
      brokerBacked ? position.broker_take_profit : position.take_profit,
    );

    if (entry != null) {
      levels.push({
        key: `${source}:${identity}:entry`,
        kind: 'entry',
        price: entry,
        title: label,
        direction: position.direction,
        source,
      });
    }
    if (stopLoss != null) {
      levels.push({
        key: `${source}:${identity}:sl`,
        kind: 'stop_loss',
        price: stopLoss,
        title: `${protectionLabelPrefix}SL #${identity}`,
        direction: position.direction,
        source,
      });
    }
    if (takeProfit != null) {
      levels.push({
        key: `${source}:${identity}:tp`,
        kind: 'take_profit',
        price: takeProfit,
        title: `${protectionLabelPrefix}TP #${identity}`,
        direction: position.direction,
        source,
      });
    }
  }

  return levels;
}
