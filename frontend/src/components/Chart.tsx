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
import { getCandles, type Candle } from '../services/api';
import type { AnalysisResult } from '../types/analysis';

// Define the props interface for the component
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

  // Fetch candle data
  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    getCandles(symbol, timeframe)
      .then(response => {
        if (!cancelled) {
          setCandles(response.candles);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message);
          setCandles([]);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [symbol, timeframe]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || chartRef.current) {
      return;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: { background: { color: '#0b0b0f' }, textColor: '#dfe7ff' },
      grid: { vertLines: { color: '#242b3a' }, horzLines: { color: '#242b3a' } },
      timeScale: { timeVisible: true, secondsVisible: false, rightOffset: 4 },
      rightPriceScale: { borderColor: '#2b3350' },
      crosshair: { mode: 0 },
    });
    chartRef.current = chart;

    candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#58d68d',
      downColor: '#ff6b6b',
      wickUpColor: '#58d68d',
      wickDownColor: '#ff6b6b',
      borderVisible: false,
    });
    setChartReady(true);

    const resizeObserver = new ResizeObserver(entries => {
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
    if (!chartReady || !candleSeriesRef.current) {
      return;
    }

    if (candles.length > 0) {
      const seriesData: CandlestickData<Time>[] = candles.map(candle => ({
        time: candle.time as Time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }));
      candleSeriesRef.current.setData(seriesData);
      chartRef.current?.timeScale().fitContent();
    } else if (!loading) {
      candleSeriesRef.current.setData([]);
    }
  }, [candles, loading, chartReady]);

  // Draw analysis lines
  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) {
      return;
    }

    // Clear previous lines
    priceLinesRef.current.forEach(line => candleSeriesRef.current?.removePriceLine(line));
    priceLinesRef.current = [];

    if (analysis) {
      const { entry, tp, take_profit, sl, stop_loss, signal } = analysis;
      const color = signal === 'long' ? '#58d68d' : '#ff6b6b';
      const tpPrice = tp ?? take_profit;
      const slPrice = sl ?? stop_loss;

      const createLine = (price: number, label: string, lineStyle: LineStyle = LineStyle.Dotted) => {
        const line = candleSeriesRef.current?.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle,
          axisLabelVisible: true,
          title: label,
        });
        if (line) {
          priceLinesRef.current.push(line);
        }
      };

      if (entry) {
        createLine(entry, 'Entry', LineStyle.Solid);
      }
      if (tpPrice) {
        createLine(tpPrice, 'TP');
      }
      if (slPrice) {
        createLine(slPrice, 'SL');
      }
    }
  }, [analysis, chartReady]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={chartContainerRef} id="chart" style={{ width: '100%', height: '100%' }} />
      {(loading || error || (!loading && candles.length === 0)) && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#0b0b0fcc',
            color: '#dfe7ff',
            zIndex: 1,
            fontSize: '1rem',
          }}
        >
          {loading ? 'Loading…' : error ? `Error: ${error}` : 'No data available'}
        </div>
      )}
    </div>
  );
}
