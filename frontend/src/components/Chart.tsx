import { useEffect, useMemo, useRef, useState } from 'react';
import {
  createChart,
  CandlestickSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  type IPriceLine,
  LineStyle,
} from 'lightweight-charts';
import { getV2Candles, type Candle } from '../services/api';
import type { AnalysisResult } from '../types/analysis';

const CHART_CANDLE_LIMIT = 1500;
const LIVE_REFRESH_MS: Record<string, number> = {
  M1: 1000,
  M5: 2000,
  M15: 5000,
  M30: 10000,
  H1: 15000,
  H4: 30000,
  D1: 60000,
};

interface ChartProps {
  symbol: string;
  timeframe: string;
  analysis: AnalysisResult | null;
}

function refreshIntervalForTimeframe(timeframe: string): number {
  return LIVE_REFRESH_MS[timeframe.toUpperCase()] ?? 5000;
}

function sameCandles(a: Candle[], b: Candle[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  if (a.length === 0) return true;
  const lastA = a[a.length - 1];
  const lastB = b[b.length - 1];
  const firstA = a[0];
  const firstB = b[0];
  return (
    firstA.time === firstB.time
    && lastA.time === lastB.time
    && lastA.open === lastB.open
    && lastA.high === lastB.high
    && lastA.low === lastB.low
    && lastA.close === lastB.close
  );
}

function chartErrorMessage(message: string): string {
  if (message.toLowerCase().includes('not authorized')) {
    return 'cTrader account is not authorized. Reconnect or refresh account authorization before loading live candles.';
  }
  return message;
}

export default function Chart({ symbol, timeframe, analysis }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartReady, setChartReady] = useState(false);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const autoFitKeyRef = useRef('');
  const refreshMs = useMemo(() => refreshIntervalForTimeframe(timeframe), [timeframe]);
  const chartKey = `${symbol}:${timeframe}`;

  useEffect(() => {
    autoFitKeyRef.current = '';
  }, [chartKey]);

  useEffect(() => {
    let disposed = false;
    let timer: number | null = null;
    let controller: AbortController | null = null;

    const scheduleNext = () => {
      if (disposed) return;
      timer = window.setTimeout(fetchCandles, refreshMs);
    };

    const fetchCandles = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const response = await getV2Candles(symbol, timeframe, CHART_CANDLE_LIMIT, controller.signal, { live: true });
        if (disposed) return;
        setCandles((prev) => (sameCandles(prev, response.candles) ? prev : response.candles));
        setError(null);
      } catch (err) {
        if (disposed) return;
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        setError(err instanceof Error ? chartErrorMessage(err.message) : 'Failed to refresh chart');
      } finally {
        if (!disposed) {
          setLoading(false);
          scheduleNext();
        }
      }
    };

    setLoading(true);
    setError(null);
    setCandles([]);
    fetchCandles();

    return () => {
      disposed = true;
      controller?.abort();
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [symbol, timeframe, refreshMs]);

  useEffect(() => {
    if (!chartContainerRef.current || chartRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#080b12' },
        textColor: '#5a6a8a',
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(99, 102, 241, 0.04)' },
        horzLines: { color: 'rgba(99, 102, 241, 0.04)' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        borderColor: 'rgba(99, 102, 241, 0.08)',
      },
      rightPriceScale: {
        borderColor: 'rgba(99, 102, 241, 0.08)',
      },
      crosshair: {
        mode: 0,
        vertLine: { color: 'rgba(99, 102, 241, 0.3)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#6366f1' },
        horzLine: { color: 'rgba(99, 102, 241, 0.3)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#6366f1' },
      },
    });
    chartRef.current = chart;

    candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
      borderVisible: false,
    });
    setChartReady(true);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        chart.resize(width, height);
      }
    });
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      setChartReady(false);
    };
  }, []);

  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) return;

    if (candles.length > 0) {
      const seriesData: CandlestickData<Time>[] = candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));
      candleSeriesRef.current.setData(seriesData);
      if (autoFitKeyRef.current !== chartKey) {
        chartRef.current?.timeScale().fitContent();
        autoFitKeyRef.current = chartKey;
      }
    } else if (!loading) {
      candleSeriesRef.current.setData([]);
    }
  }, [candles, loading, chartReady, chartKey]);

  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) return;

    priceLinesRef.current.forEach((line) => candleSeriesRef.current?.removePriceLine(line));
    priceLinesRef.current = [];

    if (analysis) {
      const { entry, tp, sl } = analysis;

      const createLine = (price: number, label: string, color: string, style: LineStyle = LineStyle.Dotted) => {
        const line = candleSeriesRef.current?.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: style,
          axisLabelVisible: true,
          title: label,
        });
        if (line) priceLinesRef.current.push(line);
      };

      if (entry) createLine(entry, 'Entry', '#8b5cf6', LineStyle.Solid);
      if (tp) createLine(tp, 'TP', '#10b981');
      if (sl) createLine(sl, 'SL', '#ef4444');
    }
  }, [analysis, chartReady]);

  const showOverlay = candles.length === 0 && (loading || !!error || !loading);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
      {showOverlay && (
        <div className="ta-chart-loading">
          {loading ? (
            <>
              <div className="ta-spinner" />
              <div className="ta-chart-loading__text">Loading {symbol} {timeframe}...</div>
            </>
          ) : error ? (
            <div className="ta-chart-loading__text" style={{ color: 'var(--ta-bear)' }}>
              Error: {error}
            </div>
          ) : (
            <div className="ta-chart-loading__text">No data available</div>
          )}
        </div>
      )}
    </div>
  );
}
