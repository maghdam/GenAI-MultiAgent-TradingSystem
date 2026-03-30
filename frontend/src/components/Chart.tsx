import { useEffect, useRef, useState } from 'react';
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

interface ChartProps {
  symbol: string;
  timeframe: string;
  analysis: AnalysisResult | null;
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

  /* Fetch candle data */
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    getV2Candles(symbol, timeframe, CHART_CANDLE_LIMIT, controller.signal)
      .then((response) => {
        setCandles(response.candles);
        setLoading(false);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        setError(err.message);
        setCandles([]);
        setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [symbol, timeframe]);

  /* Initialize chart */
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

  /* Set candle data */
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
      chartRef.current?.timeScale().fitContent();
    } else if (!loading) {
      candleSeriesRef.current.setData([]);
    }
  }, [candles, loading, chartReady]);

  /* Draw analysis lines */
  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) return;

    priceLinesRef.current.forEach((line) => candleSeriesRef.current?.removePriceLine(line));
    priceLinesRef.current = [];

    if (analysis) {
      const { entry, tp, sl, signal } = analysis;

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

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
      {(loading || error || (!loading && candles.length === 0)) && (
        <div className="ta-chart-loading">
          {loading ? (
            <>
              <div className="ta-spinner" />
              <div className="ta-chart-loading__text">Loading {symbol} {timeframe}…</div>
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
