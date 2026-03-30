import React, { useMemo } from 'react';

import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Legend, ComposedChart, Line, LineChart, Scatter
} from 'recharts';
import { ResizableBox } from './ResizableBox';

interface Trade {
    entry_time: string;
    exit_time: string;
    entry_price: number;
    exit_price: number;
    pnl: number;
    return_pct: number;
    direction: string;
}

interface Candle {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface BacktestData {
    metrics: Record<string, string | number>;
    equity: { time: string; value: number }[];
    trades: Trade[];
    candles?: Candle[]; // Optional for now
    optimization_results?: any[];
    best_result?: any;
    plots?: string; // JSON string of plotly figure
}

interface Props {
    data: BacktestData;
    resizable?: boolean;
}

interface PlotTrace {
    name?: string;
    type?: string;
    mode?: string;
    x?: unknown[];
    y?: unknown[];
    line?: { color?: string };
    marker?: { color?: string };
}

interface PlotPayload {
    data?: PlotTrace[];
}

const TRACE_COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#ba68c8', '#e57373', '#90a4ae'];

// Custom markers for buy/sell
const BuyMarker = (props: any) => {
    const { cx, cy } = props;
    return (
        <path d={`M${cx - 5},${cy + 5} L${cx + 5},${cy + 5} L${cx},${cy - 5} Z`} fill="#4caf50" stroke="none" />
    );
};

const SellMarker = (props: any) => {
    const { cx, cy } = props;
    return (
        <path d={`M${cx - 5},${cy - 5} L${cx + 5},${cy - 5} L${cx},${cy + 5} Z`} fill="#f44336" stroke="none" />
    );
};

function formatMetricValue(key: string, value: string | number): string {
    if (typeof value !== 'number') return String(value);
    if (!Number.isFinite(value)) return '';

    const k = key.toLowerCase();
    if (key.includes('[%]')) return `${value.toFixed(2)}%`;
    if (k.includes('ratio') || k.includes('factor')) return value.toFixed(2);
    if (k.includes('fees') || k.includes('pnl') || k.includes('expectancy') || k.includes('value')) return value.toFixed(2);

    const abs = Math.abs(value);
    if (abs >= 1) return value.toFixed(2);
    return value.toFixed(4);
}

function valueColor(key: string, value: string | number): string | undefined {
    if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
    const k = key.toLowerCase();

    // Avoid misleading coloring for risk/cost metrics.
    if (k.includes('drawdown') || k.includes('exposure') || k.includes('fees')) return undefined;

    const signed =
        k.includes('return') ||
        k.includes('pnl') ||
        k.includes('expectancy') ||
        k.includes('sharpe') ||
        k.includes('sortino') ||
        k.includes('calmar') ||
        k.includes('best trade') ||
        k.includes('worst trade') ||
        k.includes('avg winning trade') ||
        k.includes('avg losing trade');

    if (!signed) return undefined;
    if (value > 0) return 'var(--good)';
    if (value < 0) return 'var(--bad)';
    return undefined;
}

function formatPlotLabel(value: unknown, index: number): string {
    if (typeof value === 'string' && value.trim()) {
        return value;
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
        return String(value);
    }
    return `#${index + 1}`;
}

function buildPlotRows(payload: PlotPayload | null): {
    rows: Array<Record<string, string | number | null>>;
    traces: Array<{ key: string; label: string; color: string }>;
} {
    if (!payload?.data?.length) {
        return { rows: [], traces: [] };
    }

    const plottable = payload.data
        .filter((trace) => Array.isArray(trace?.y) && trace.y.some((value) => typeof value === 'number' && Number.isFinite(value)))
        .slice(0, 6);

    if (!plottable.length) {
        return { rows: [], traces: [] };
    }

    const maxLength = plottable.reduce((acc, trace) => Math.max(acc, Array.isArray(trace.y) ? trace.y.length : 0), 0);
    const step = Math.max(1, Math.ceil(maxLength / 300));
    const traces = plottable.map((trace, index) => ({
        key: `trace_${index}`,
        label: trace.name || `Trace ${index + 1}`,
        color: trace.line?.color || trace.marker?.color || TRACE_COLORS[index % TRACE_COLORS.length],
    }));

    const rows: Array<Record<string, string | number | null>> = [];
    for (let rawIndex = 0; rawIndex < maxLength; rawIndex += step) {
        const firstTrace = plottable[0];
        const label = formatPlotLabel(firstTrace?.x?.[rawIndex], rawIndex);
        const row: Record<string, string | number | null> = {
            label,
            index: rawIndex + 1,
        };
        traces.forEach((traceMeta, traceIndex) => {
            const source = plottable[traceIndex];
            const value = source?.y?.[rawIndex];
            row[traceMeta.key] = typeof value === 'number' && Number.isFinite(value) ? value : null;
        });
        rows.push(row);
    }
    return { rows, traces };
}

export function BacktestDashboard({ data, resizable = false }: Props) {
    const { metrics, equity, trades, optimization_results, candles } = data;

    const hasEquity = equity && equity.length > 0;
    const hasOpt = Boolean(optimization_results && optimization_results.length > 0);
    const hasCandles = candles && candles.length > 0;

    // Merge candles and trades for Price Chart
    const priceChartData = useMemo(() => {
        if (!hasCandles && !hasEquity) return [];

        // If we have candles, use them as base
        const base = hasCandles ? candles! : equity.map(e => ({ time: e.time, close: 0 })); // Fallback if no candles, but unlikely if valid backtest

        return base.map(c => {
            // Find entries at this candle
            // Note: String matching can be fragile with timezones, ensuring simple string match from backend
            const entryLong = trades.find(t => t.entry_time === c.time && t.direction === 'Long');
            const entryShort = trades.find(t => t.entry_time === c.time && t.direction === 'Short');

            const exitLong = trades.find(t => t.exit_time === c.time && t.direction === 'Long');
            const exitShort = trades.find(t => t.exit_time === c.time && t.direction === 'Short');

            // Prioritize showing Entries over Exits if conflict, or show both if possible?
            // For simplicity in ComposedChart, we'll just show markers for Entry Long (Green Up) and Entry Short (Red Down)? 
            // Or standard VBT style: Entry Long (Green ^), Exit Long (Black x), Entry Short (Red v).

            // Let's simplify: 
            // Buy Signal: Entry Long OR Exit Short
            // Sell Signal: Entry Short OR Exit Long
            // Actually, let's just mark ENTRIES for now to reduce clutter, as exits are often stoploss/tp

            return {
                time: c.time,
                close: hasCandles ? c.close : undefined,
                equity: hasEquity ? equity.find(e => e.time === c.time)?.value : undefined,

                entry_long: entryLong ? entryLong.entry_price : null,
                entry_short: entryShort ? entryShort.entry_price : null,

                exit_long: exitLong ? exitLong.exit_price : null,
                exit_short: exitShort ? exitShort.exit_price : null,
            };
        });
    }, [candles, equity, trades, hasCandles, hasEquity]);

    const plotData = useMemo<PlotPayload | null>(() => {
        if (!data.plots) return null;
        try {
            return JSON.parse(data.plots);
        } catch (e) {
            console.error("Failed to parse plots JSON", e);
            return null;
        }
    }, [data.plots]);
    const plotSeries = useMemo(() => buildPlotRows(plotData), [plotData]);

    const metricGroups = useMemo(() => {
        const used = new Set<string>();
        const metaKeys = new Set(['Start', 'End', 'Period', 'Start Value', 'End Value']);

        const take = (keys: string[]) => {
            const out: Array<[string, string | number]> = [];
            for (const k of keys) {
                const v = metrics[k];
                if (v === undefined) continue;
                used.add(k);
                out.push([k, v]);
            }
            return out;
        };

        const groups = [
            { title: 'Performance', items: take([
                'Total Return [%]',
                'Benchmark Return [%]',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Calmar Ratio',
                'Omega Ratio',
                'Max Drawdown [%]',
                'Max Drawdown Duration',
            ]) },
            { title: 'Trades', items: take([
                'Total Trades',
                'Total Closed Trades',
                'Total Open Trades',
                'Win Rate [%]',
                'Profit Factor',
                'Expectancy',
                'Best Trade [%]',
                'Worst Trade [%]',
                'Avg Winning Trade [%]',
                'Avg Losing Trade [%]',
                'Avg Winning Trade Duration',
                'Avg Losing Trade Duration',
                'Open Trade PnL',
            ]) },
            { title: 'Costs & Exposure', items: take([
                'Total Fees Paid',
                'Max Gross Exposure [%]',
            ]) },
        ];

        const otherKeys = Object.keys(metrics)
            .filter((k) => !metaKeys.has(k) && !used.has(k))
            .sort((a, b) => a.localeCompare(b));

        const otherItems: Array<[string, string | number]> = [];
        for (const k of otherKeys) {
            const v = metrics[k];
            if (v === undefined) continue;
            otherItems.push([k, v]);
        }

        if (otherItems.length) groups.push({ title: 'Other', items: otherItems });
        return groups.filter((g) => g.items.length > 0);
    }, [metrics]);

    const metaLine = useMemo(() => {
        const start = metrics['Start'];
        const end = metrics['End'];
        const period = metrics['Period'];
        const startValue = metrics['Start Value'];
        const endValue = metrics['End Value'];

        const parts: string[] = [];
        if (start && end) parts.push(`${start} → ${end}`);
        if (period) parts.push(String(period));
        if (startValue !== undefined && endValue !== undefined) {
            parts.push(`${formatMetricValue('Start Value', startValue)} → ${formatMetricValue('End Value', endValue)}`);
        }
        return parts.join(' · ');
    }, [metrics]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* 1. Metrics Report Table (User Preferred Style) */}
            <ResizableBox
                className="box"
                storageKey={resizable ? 'strategyStudio.bt.metrics' : undefined}
                defaultHeight={260}
                minHeight={180}
                enable={resizable ? ['s', 'se'] : []}
                style={{ padding: '15px', display: 'flex', flexDirection: 'column' }}
            >
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                    <h3 style={{ marginTop: 0, marginBottom: 8 }}>Backtest Report</h3>
                    {metaLine ? <div className="muted" style={{ fontSize: 12 }}>{metaLine}</div> : null}
                </div>
                <div style={{ flex: 1, minHeight: 0, overflow: 'auto', paddingRight: 4 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 14 }}>
                        {metricGroups.map((g) => (
                            <div key={g.title} style={{ border: '1px solid #171a24', borderRadius: 10, padding: 10 }}>
                                <div style={{ fontWeight: 700, marginBottom: 8 }}>{g.title}</div>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9em' }}>
                                    <tbody>
                                        {g.items.map(([k, v]) => {
                                            const formatted = formatMetricValue(k, v);
                                            const color = valueColor(k, v);
                                            return (
                                                <tr key={k}>
                                                    <td style={{ padding: '6px 0', color: 'var(--muted)' }}>{k}</td>
                                                    <td
                                                        style={{ padding: '6px 0', textAlign: 'right', fontWeight: 700, color }}
                                                    >
                                                        {formatted}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        ))}
                    </div>
                </div>
            </ResizableBox>

            {/* 2. Simplified VectorBT Trace View */}
            {plotSeries.rows.length ? (
                <ResizableBox
                    className="box"
                    storageKey={resizable ? 'strategyStudio.bt.vbtChart' : undefined}
                    defaultHeight={420}
                    minHeight={280}
                    enable={resizable ? ['e', 's', 'se'] : []}
                    style={{ padding: '10px' }}
                >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', height: '100%' }}>
                        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                            <h3 style={{ margin: 0 }}>VectorBT Trace View</h3>
                            <div className="muted" style={{ fontSize: '12px' }}>
                                Simplified renderer using the first {plotSeries.traces.length} numeric traces.
                            </div>
                        </div>
                        <div style={{ flex: 1, minHeight: 0 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={plotSeries.rows}>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                    <XAxis dataKey="label" tick={{ fontSize: 10 }} minTickGap={30} />
                                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #333' }}
                                        itemStyle={{ color: '#ccc' }}
                                        labelStyle={{ color: '#888' }}
                                    />
                                    <Legend />
                                    {plotSeries.traces.map((trace) => (
                                        <Line
                                            key={trace.key}
                                            type="monotone"
                                            dataKey={trace.key}
                                            name={trace.label}
                                            stroke={trace.color}
                                            dot={false}
                                            strokeWidth={2}
                                            connectNulls
                                        />
                                    ))}
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </ResizableBox>
            ) : plotData ? (
                <div className="box" style={{ padding: '20px', display: 'grid', gap: '12px' }}>
                    <div>
                        <h3 style={{ margin: 0 }}>VectorBT Trace View</h3>
                    </div>
                    <div className="muted">
                        Plot data was present, but it did not contain numeric traces that the lightweight renderer could plot.
                    </div>
                    <div className="muted" style={{ fontSize: '12px' }}>
                        The metrics, price chart, equity curve, optimization table, and trades list below remain available.
                    </div>
                </div>
            ) : (
                <div className="box" style={{ padding: '20px', display: 'grid', gap: '12px' }}>
                    <div>
                        <h3 style={{ margin: 0 }}>VectorBT Trace View</h3>
                    </div>
                    <div className="muted">
                        No advanced trace payload was returned for this result.
                    </div>
                    <div className="muted" style={{ fontSize: '12px' }}>
                        The rest of the backtest report still renders from the structured result data.
                    </div>
                </div>
            )}

            {/* 3. Trades List (Optional, can keep existing or remove if Plotly is enough) */}
            {/* Kept as separate component or block if needed */}

            {/* 2. Optimization Heatmap / Table (if applicable) */}
            {hasOpt && (
                <div className="box">
                    <h3>Optimization Results</h3>
                    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                        <table className="journal-table" style={{ width: '100%' }}>
                            <thead>
                                <tr>
                                    <th>Params</th>
                                    <th>Return %</th>
                                    <th>Sharpe</th>
                                    <th>Drawdown %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {optimization_results!.slice(0, 50).map((res, i) => ( // limit to 50
                                    <tr key={i} style={{ backgroundColor: i === 0 ? 'rgba(76, 175, 80, 0.1)' : undefined }}>
                                        <td>{res.params || res.Param || `${res.Fast}/${res.Slow}`}</td>
                                        <td>{res["Total Return [%]"]?.toFixed(2)}</td>
                                        <td>{res["Sharpe Ratio"]?.toFixed(2)}</td>
                                        <td>{res["Max Drawdown [%]"]?.toFixed(2)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* 3. Price Chart with Overlays */}
            {hasCandles && (
                <ResizableBox
                    className="box"
                    storageKey={resizable ? 'strategyStudio.bt.priceChart' : undefined}
                    defaultHeight={400}
                    minHeight={260}
                    enable={resizable ? ['e', 's', 'se'] : []}
                    style={{ display: 'flex', flexDirection: 'column' }}
                >
                    <h3>Price & Trades</h3>
                    <div style={{ flex: 1, minHeight: 0 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={priceChartData}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                <XAxis
                                    dataKey="time"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(val) => val.split(' ')[0]}
                                    minTickGap={30}
                                />
                                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #333' }}
                                    itemStyle={{ color: '#ccc' }}
                                    labelStyle={{ color: '#888' }}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="close" stroke="#8884d8" dot={false} strokeWidth={2} name="Price" />

                                {/* Markers */}
                                <Scatter name="Entry Long" dataKey="entry_long" shape={<BuyMarker />} fill="#4caf50" />
                                <Scatter name="Entry Short" dataKey="entry_short" shape={<SellMarker />} fill="#f44336" />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </ResizableBox>
            )}

            {/* 4. Equity Curve */}
            {hasEquity && (
                <ResizableBox
                    className="box"
                    storageKey={resizable ? 'strategyStudio.bt.equityChart' : undefined}
                    defaultHeight={300}
                    minHeight={220}
                    enable={resizable ? ['e', 's', 'se'] : []}
                    style={{ display: 'flex', flexDirection: 'column' }}
                >
                    <h3>Equity Curve</h3>
                    <div style={{ flex: 1, minHeight: 0 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={equity}>
                                <defs>
                                    <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#2196f3" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#2196f3" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                <XAxis
                                    dataKey="time"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(val) => val.split(' ')[0]}
                                    minTickGap={30}
                                />
                                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #333' }}
                                    itemStyle={{ color: '#ccc' }}
                                    labelStyle={{ color: '#888' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#2196f3"
                                    fillOpacity={1}
                                    fill="url(#colorVal)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </ResizableBox>
            )}

            {/* 5. Trades List */}
            <div className="box">
                <h3>Trades ({trades?.length || 0})</h3>
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    <table className="journal-table" style={{ width: '100%' }}>
                        <thead>
                            <tr>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>Dir</th>
                                <th>Price In</th>
                                <th>Price Out</th>
                                <th>PnL</th>
                                <th>Ret %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades?.slice().reverse().map((t, i) => (
                                <tr key={i}>
                                    <td style={{ fontSize: '0.8em' }}>{t.entry_time.replace('T', ' ')}</td>
                                    <td style={{ fontSize: '0.8em' }}>{t.exit_time.replace('T', ' ')}</td>
                                    <td>
                                        <span style={{
                                            color: t.direction === 'Long' ? '#4caf50' : '#f44336',
                                            fontWeight: 600,
                                            fontSize: '0.8em',
                                            padding: '2px 6px',
                                            borderRadius: '4px',
                                            backgroundColor: t.direction === 'Long' ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)'
                                        }}>
                                            {t.direction}
                                        </span>
                                    </td>
                                    <td>{t.entry_price.toFixed(5)}</td>
                                    <td>{t.exit_price.toFixed(5)}</td>
                                    <td style={{ color: t.pnl >= 0 ? '#4caf50' : '#f44336' }}>
                                        {t.pnl.toFixed(2)}
                                    </td>
                                    <td style={{ color: t.return_pct >= 0 ? '#4caf50' : '#f44336', fontWeight: 600 }}>
                                        {t.return_pct.toFixed(2)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    );
}
