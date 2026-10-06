import { useEffect, useMemo, useRef, useState } from 'react';
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  type IPriceLine,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  LineStyle,
} from 'lightweight-charts';
import { getV2Candles, type Candle, type V2Analysis, type V2PaperPosition } from '../services/api';
import { buildChartPositionLevels } from '../services/chartPositionLines';
import { buildChartSignalMarkers, selectedSignalToAnalysis } from '../services/chartSignals';
import type { AgentSignal } from '../types';
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
  positions: V2PaperPosition[];
  signals: V2Analysis[];
  selectedSignal: AgentSignal | null;
  tradeQuantity: number;
  onTradeSelectedSignal: () => void;
  onClearSelectedSignal: () => void;
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

function formatSignalPrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '–';
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 2 : abs >= 1 ? 4 : 6;
  return value.toFixed(digits);
}

function formatSignalTime(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) return 'Unknown time';
  return new Date(timestamp * 1000).toLocaleString();
}

export default function Chart({
  symbol,
  timeframe,
  analysis,
  positions,
  signals,
  selectedSignal,
  tradeQuantity,
  onTradeSelectedSignal,
  onClearSelectedSignal,
}: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const signalMarkersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartReady, setChartReady] = useState(false);
  const [tradeReviewOpen, setTradeReviewOpen] = useState(false);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const autoFitKeyRef = useRef('');
  const refreshMs = useMemo(() => refreshIntervalForTimeframe(timeframe), [timeframe]);
  const positionLevels = useMemo(
    () => buildChartPositionLevels(positions, symbol),
    [positions, symbol],
  );
  const signalMarkers = useMemo(
    () => buildChartSignalMarkers(signals, candles, symbol, timeframe),
    [signals, candles, symbol, timeframe],
  );
  const displayedAnalysis = useMemo(
    () => selectedSignal ? selectedSignalToAnalysis(selectedSignal) : analysis,
    [selectedSignal, analysis],
  );
  const chartKey = `${symbol}:${timeframe}`;

  useEffect(() => {
    autoFitKeyRef.current = '';
  }, [chartKey]);

  useEffect(() => {
    setTradeReviewOpen(false);
  }, [selectedSignal?.ts, selectedSignal?.symbol, selectedSignal?.timeframe, selectedSignal?.strategy]);

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
    signalMarkersRef.current = createSeriesMarkers(candleSeriesRef.current, []);
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
      signalMarkersRef.current?.detach();
      signalMarkersRef.current = null;
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
    if (!chartReady || !signalMarkersRef.current) return;
    const markers: SeriesMarker<Time>[] = signalMarkers.map((marker) => ({
      id: marker.id,
      time: marker.time as Time,
      position: marker.position,
      shape: marker.shape,
      color: marker.color,
      text: marker.text,
      size: 1,
    }));
    signalMarkersRef.current.setMarkers(markers);
  }, [chartReady, signalMarkers]);

  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) return;

    priceLinesRef.current.forEach((line) => candleSeriesRef.current?.removePriceLine(line));
    priceLinesRef.current = [];

    const createLine = (
      price: number,
      label: string,
      color: string,
      style: LineStyle = LineStyle.Dotted,
      lineWidth: 1 | 2 = 1,
    ) => {
      const line = candleSeriesRef.current?.createPriceLine({
        price,
        color,
        lineWidth,
        lineStyle: style,
        axisLabelVisible: true,
        title: label,
      });
      if (line) priceLinesRef.current.push(line);
    };

    if (displayedAnalysis) {
      const { entry, tp, sl } = displayedAnalysis;
      const labelPrefix = selectedSignal ? 'Selected Signal' : 'Signal';
      if (entry != null && Number.isFinite(entry)) {
        createLine(entry, `${labelPrefix} Entry`, '#8b5cf6', LineStyle.Solid);
      }
      if (tp != null && Number.isFinite(tp)) {
        createLine(tp, `${labelPrefix} TP`, '#10b981');
      }
      if (sl != null && Number.isFinite(sl)) {
        createLine(sl, `${labelPrefix} SL`, '#ef4444');
      }
    }

    for (const level of positionLevels) {
      if (level.kind === 'entry') {
        createLine(
          level.price,
          level.title,
          level.direction === 'long' ? '#38bdf8' : '#f59e0b',
          LineStyle.Solid,
          2,
        );
      } else if (level.kind === 'stop_loss') {
        createLine(level.price, level.title, '#ef4444', LineStyle.Dashed);
      } else {
        createLine(level.price, level.title, '#10b981', LineStyle.Dashed);
      }
    }
  }, [displayedAnalysis, selectedSignal, chartReady, positionLevels]);

  const showOverlay = candles.length === 0 && (loading || !!error || !loading);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />
      {selectedSignal && (
        <div className="ta-chart-signal-card" role="region" aria-label="Selected trading signal">
          <div className="ta-chart-signal-card__header">
            <span className={`ta-pill ${selectedSignal.signal === 'long' ? 'ta-pill--long' : selectedSignal.signal === 'short' ? 'ta-pill--short' : 'ta-pill--no_trade'}`}>
              {selectedSignal.signal === 'long' ? '▲ BUY' : selectedSignal.signal === 'short' ? '▼ SELL' : 'NO TRADE'}
            </span>
            <strong>{selectedSignal.symbol} · {selectedSignal.timeframe}</strong>
            <button
              type="button"
              className="ta-chart-signal-card__close"
              onClick={onClearSelectedSignal}
              aria-label="Clear selected signal"
              title="Clear selected signal"
            >
              ×
            </button>
          </div>

          <div className="ta-chart-signal-card__meta">
            <span>{selectedSignal.strategy.replace(/_/g, ' ')}</span>
            <span>Strength {Math.round(selectedSignal.confidence * 100)}%</span>
            <span>{formatSignalTime(selectedSignal.ts)}</span>
          </div>

          <div className="ta-chart-signal-card__levels">
            <span>Entry <strong>{formatSignalPrice(selectedSignal.entry)}</strong></span>
            <span>SL <strong>{formatSignalPrice(selectedSignal.sl)}</strong></span>
            <span>TP <strong>{formatSignalPrice(selectedSignal.tp)}</strong></span>
          </div>

          {selectedSignal.reasons.length > 0 && (
            <div className="ta-chart-signal-card__reason">
              {selectedSignal.reasons.slice(0, 2).join(' · ')}
            </div>
          )}

          {(selectedSignal.signal === 'long' || selectedSignal.signal === 'short') && (
            tradeReviewOpen ? (
              <div className="ta-chart-signal-card__confirm">
                <div>
                  Submit {selectedSignal.signal.toUpperCase()} {tradeQuantity.toFixed(2)} lots through the normal risk,
                  account, protection, and Live-arm gates.
                </div>
                <div className="ta-chart-signal-card__actions">
                  <button
                    type="button"
                    className="ta-btn ta-btn--success ta-btn--sm"
                    onClick={() => {
                      setTradeReviewOpen(false);
                      onTradeSelectedSignal();
                    }}
                  >
                    Confirm order
                  </button>
                  <button
                    type="button"
                    className="ta-btn ta-btn--sm"
                    onClick={() => setTradeReviewOpen(false)}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                type="button"
                className="ta-btn ta-btn--primary ta-btn--sm ta-chart-signal-card__trade"
                onClick={() => setTradeReviewOpen(true)}
              >
                Review trade signal
              </button>
            )
          )}
        </div>
      )}
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
