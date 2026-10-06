import { useEffect, useRef, useState } from 'react';

import type { AIOutputHandle } from '../components/AIOutput';
import type { AnalysisResult } from '../types/analysis';
import {
  getV2CTraderAccounts,
  getV2Strategies,
  getV2Status,
  selectV2CTraderAccount,
  setV2Config,
  startV2Engine,
  stopV2Engine,
  type V2CTraderAccount,
  type V2StrategyInfo,
  type V2Status,
  type V2WatchlistItem,
} from '../services/api';
import type { AgentSignal } from '../types';
import Header, { type DashboardEngineStatus } from '../components/Header';
import TradeSettings from '../components/TradeSettings';
import Chart from '../components/Chart';
import SidePanel from '../components/SidePanel';
import Journal from '../components/Journal';
import AIOutput from '../components/AIOutput';
import SymbolSelector from '../components/SymbolSelector';
import MarketContextPanel from '../components/MarketContextPanel';
import {
  loadFrontendRestartSnapshot,
  type FrontendRestartSnapshot,
} from '../services/frontendRestart';

export default function DashboardPage() {
  const [symbol, setSymbol] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('M5');
  const [strategy, setStrategy] = useState('sma_cross');
  const [lotSize, setLotSize] = useState(0.01);
  const [fastMode, setFastMode] = useState(true);
  const [maxBars, setMaxBars] = useState(250);
  const [maxTokens, setMaxTokens] = useState(256);
  const [modelName, setModelName] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAgentSettingsOpen, setIsAgentSettingsOpen] = useState(false);
  const [status, setStatus] = useState<V2Status | null>(null);
  const [ctraderAccounts, setCtraderAccounts] = useState<V2CTraderAccount[]>([]);
  const [accountBusy, setAccountBusy] = useState(false);
  const [accountError, setAccountError] = useState('');
  const [selectedSignal, setSelectedSignal] = useState<AgentSignal | null>(null);
  const [v2Strategies, setV2Strategies] = useState<V2StrategyInfo[]>([]);
  const [bootstrapError, setBootstrapError] = useState('');

  const aiOutputRef = useRef<AIOutputHandle>(null);
  const initialSyncPendingRef = useRef(true);
  const selectedContextRef = useRef({ symbol, timeframe, strategy });
  selectedContextRef.current = { symbol, timeframe, strategy };

  const applyDashboardSnapshot = (
    snapshot: FrontendRestartSnapshot,
    syncSelection: boolean,
  ) => {
    if (snapshot.strategies !== null) {
      setV2Strategies(snapshot.strategies);
    }
    if (snapshot.status !== null) {
      setStatus(snapshot.status);
    }
    setBootstrapError(snapshot.errors.join(' · '));

    if (!syncSelection || snapshot.status === null || snapshot.strategies === null) {
      return;
    }

    const nextStatus = snapshot.status;
    const strategies = snapshot.strategies;
    const currentContext = selectedContextRef.current;
    const currentItem = nextStatus.config.watchlist.find(
      (item) => item.symbol === currentContext.symbol && item.timeframe === currentContext.timeframe,
    );
    if (currentItem) {
      setStrategy(currentItem.strategy);
      setLotSize(currentItem.lot_size ?? nextStatus.config.paper_trade_size);
    } else {
      setLotSize((current) => (current === 0.01 ? nextStatus.config.paper_trade_size || current : current));
    }
    if (!currentItem && !strategies.some((item) => item.key === currentContext.strategy) && strategies[0]?.key) {
      setStrategy(strategies[0].key);
    }
    initialSyncPendingRef.current = false;
  };

  const loadDashboardState = async (syncSelection = false) => {
    const snapshot = await loadFrontendRestartSnapshot({
      getStatus: getV2Status,
      getStrategies: getV2Strategies,
    });
    applyDashboardSnapshot(snapshot, syncSelection);
    return snapshot;
  };

  const loadCTraderAccounts = async () => {
    try {
      const rows = await getV2CTraderAccounts();
      setCtraderAccounts(rows);
      setAccountError('');
      return rows;
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : 'cTrader account directory unavailable.');
      return null;
    }
  };

  useEffect(() => {
    let disposed = false;

    const refresh = async () => {
      const snapshot = await loadFrontendRestartSnapshot({
        getStatus: getV2Status,
        getStrategies: getV2Strategies,
      });
      if (disposed) return;
      applyDashboardSnapshot(snapshot, initialSyncPendingRef.current);
      void getV2CTraderAccounts()
        .then((rows) => {
          if (disposed) return;
          setCtraderAccounts(rows);
          setAccountError('');
        })
        .catch((error) => {
          if (disposed) return;
          setAccountError(error instanceof Error ? error.message : 'cTrader account directory unavailable.');
        });
    };

    void refresh();
    const interval = window.setInterval(() => {
      void refresh();
    }, 6000);

    return () => {
      disposed = true;
      window.clearInterval(interval);
    };
  }, []);

  const strategyOptions = v2Strategies.map((item) => item.key);
  const engineStatus: DashboardEngineStatus | null = status ? {
    enabled: status.config.enabled,
    running: status.runtime.running,
    loopActive: status.runtime.loop_active,
    watchlistCount: status.config.watchlist.length,
    mode: status.mode,
  } : null;
  const getBrokerStatusInfo = () : { status: 'ok' | 'bad' | 'wait' | 'warn', label: string } => {
    if (!status) return { status: 'wait', label: 'Offline' };
    const { broker } = status;
    if (!broker.socket_connected) return { status: 'bad', label: 'Disconnected' };
    if (!broker.account_authorized) return { status: 'warn', label: 'Auth Required' };
    if (broker.symbols_loaded === 0) return { status: 'wait', label: 'Loading Symbols' };
    if (!broker.market_data_ready) return { status: 'wait', label: 'Syncing Data' };
    return { status: 'ok', label: 'Connected' };
  };

  const feedStatus = getBrokerStatusInfo();
  
  const llmStatus = {
    status: status?.runtime.ollama_ready ? 'ok' as const : 'bad' as const,
    label: status?.runtime.ollama_ready ? 'AI: Ready' : 'AI: Offline',
  };

  const handleRunAnalysis = () => {
    setSelectedSignal(null);
    setAnalysis(null);
    aiOutputRef.current?.runAnalysis();
  };
  const handleCancelAnalysis = () => { aiOutputRef.current?.cancelAnalysis(); };
  const handlePlaceTrade = () => { aiOutputRef.current?.placeTrade(); };
  const handleTradeSelectedSignal = () => { aiOutputRef.current?.placeTrade(); };
  const handleAnalysisComplete = (result: AnalysisResult | null) => { setAnalysis(result); setIsAnalyzing(false); };

  const handleToggleAgent = async () => {
    try {
      if (status?.config.enabled) { await stopV2Engine(); } else { await startV2Engine(); }
      await loadDashboardState(true);
    } catch (error) { console.error('Failed to toggle engine', error); }
  };

  const handleWatchCurrent = async () => {
    if (!status) return;
    try {
      const previous = status.config.watchlist.find((item) => item.symbol === symbol && item.timeframe === timeframe);
      const nextItem: V2WatchlistItem = {
        symbol,
        timeframe,
        strategy,
        enabled: previous?.enabled ?? true,
        trading_enabled: previous?.trading_enabled ?? false,
        lot_size: lotSize,
        params: previous?.params ?? {},
      };
      const existing = status.config.watchlist || [];
      const deduped = existing.filter((item) => !(item.symbol === symbol && item.timeframe === timeframe));
      await setV2Config({ ...status.config, paper_trade_size: lotSize, watchlist: [...deduped, nextItem] });
      await loadDashboardState(true);
    } catch (error) { console.error('Failed to add to watchlist', error); }
  };

  const handleReloadStrategies = async () => {
    try {
      await Promise.all([loadDashboardState(true), loadCTraderAccounts()]);
    } catch (error) { console.error('Failed to refresh', error); }
  };

  const handleCTraderAccountChange = async (accountId: number) => {
    if (!Number.isInteger(accountId) || accountId <= 0 || accountBusy) return;
    setAccountBusy(true);
    setAccountError('');
    try {
      await selectV2CTraderAccount(accountId);
      await Promise.all([loadDashboardState(false), loadCTraderAccounts()]);
    } catch (error) {
      setAccountError(error instanceof Error ? error.message : 'Failed to select cTrader account.');
    } finally {
      setAccountBusy(false);
    }
  };

  const handleSignalSelect = (signal: AgentSignal) => {
    const nextSymbol = signal.symbol || symbol;
    const nextTimeframe = signal.timeframe || timeframe;
    const watchItem = status?.config.watchlist.find(
      (entry) => entry.symbol === nextSymbol && entry.timeframe === nextTimeframe,
    );

    setSymbol(nextSymbol);
    setTimeframe(nextTimeframe);
    setStrategy(signal.strategy);
    setLotSize(watchItem?.lot_size ?? status?.config.paper_trade_size ?? lotSize);
    setAnalysis(null);
    setSelectedSignal(signal);
  };

  const applySymbolSettings = (nextSymbol: string, nextTimeframe: string) => {
    const item = status?.config.watchlist.find(
      (entry) => entry.symbol === nextSymbol && entry.timeframe === nextTimeframe,
    );
    if (!item) return;
    setStrategy(item.strategy);
    setLotSize(item.lot_size ?? status?.config.paper_trade_size ?? 0.01);
  };

  const clearSignalAndAnalysis = () => {
    setSelectedSignal(null);
    setAnalysis(null);
  };

  const handleStrategyChange = (nextStrategy: string) => {
    clearSignalAndAnalysis();
    setStrategy(nextStrategy);
  };

  const handleSymbolChange = (nextSymbol: string) => {
    clearSignalAndAnalysis();
    setSymbol(nextSymbol);
    applySymbolSettings(nextSymbol, timeframe);
  };

  const handleTimeframeChange = (nextTimeframe: string) => {
    clearSignalAndAnalysis();
    setTimeframe(nextTimeframe);
    applySymbolSettings(symbol, nextTimeframe);
  };

  return (
    <div className="ta-app">
      <Header
        strategy={strategy}
        strategyOptions={strategyOptions}
        onStrategyChange={handleStrategyChange}
        lotSize={lotSize}
        onLotSizeChange={setLotSize}
        fastMode={fastMode}
        onFastModeChange={setFastMode}
        maxBars={maxBars}
        onMaxBarsChange={setMaxBars}
        maxTokens={maxTokens}
        onMaxTokensChange={setMaxTokens}
        modelName={modelName}
        onModelNameChange={setModelName}
        isAnalyzing={isAnalyzing}
        onRunAnalysis={handleRunAnalysis}
        onCancelAnalysis={handleCancelAnalysis}
        onPlaceTrade={handlePlaceTrade}
        placeTradeDisabled={selectedSignal !== null || analysis === null}
        placeTradeTitle={
          selectedSignal
            ? 'Use the selected signal card on the chart to review and confirm this order.'
            : analysis
              ? 'Place trade from the current manual analysis.'
              : 'Run an analysis before placing a trade.'
        }
        feedStatus={feedStatus}
        llmStatus={llmStatus}
        onOpenSettings={() => setIsAgentSettingsOpen(true)}
        engineStatus={engineStatus}
        onToggleEngine={handleToggleAgent}
        onWatchCurrent={handleWatchCurrent}
        onRefreshStrategies={handleReloadStrategies}
        symbol={symbol}
        timeframe={timeframe}
        onTimeframeChange={handleTimeframeChange}
        ctraderAccounts={ctraderAccounts}
        ctraderAccountBusy={accountBusy}
        ctraderAccountSwitchInProgress={status?.broker.account_switch_in_progress ?? false}
        liveTradingArmed={status?.live_trading_armed ?? false}
        onCTraderAccountChange={handleCTraderAccountChange}
      />

      {/* Symbol selector row — compact */}
      <div className="ta-toolbar" style={{ paddingTop: '6px', paddingBottom: '6px', borderBottom: 'none' }}>
        <SymbolSelector onSymbolChange={handleSymbolChange} value={symbol} />
      </div>

      {bootstrapError && (
        <div className="v2-banner v2-banner-bad">
          Frontend refresh degraded: {bootstrapError}. Retaining last known backend state and retrying automatically.
        </div>
      )}

      {accountError && (
        <div className="v2-banner v2-banner-bad">
          cTrader account selection: {accountError}
        </div>
      )}

      <MarketContextPanel symbol={symbol} />

      <TradeSettings isOpen={isAgentSettingsOpen} onClose={() => setIsAgentSettingsOpen(false)} />

      {/* ─── Main Layout: Chart + Sidebar ─── */}
      <div className="ta-main">
        <div className="ta-main__chart">
          <Chart
            symbol={symbol}
            timeframe={timeframe}
            analysis={analysis}
            positions={status?.paper_positions ?? []}
            signals={status?.recent_analyses ?? []}
            selectedSignal={selectedSignal}
            tradeQuantity={lotSize}
            onTradeSelectedSignal={handleTradeSelectedSignal}
            onClearSelectedSignal={() => setSelectedSignal(null)}
          />
        </div>
        <div className="ta-main__sidebar">
          <SidePanel status={status} onSignalSelected={handleSignalSelect} />
        </div>
      </div>

      {/* ─── Bottom: Analysis + Journal ─── */}
      <div className="ta-bottom">
        <AIOutput
          hideToolbar
          ref={aiOutputRef}
          symbol={symbol}
          timeframe={timeframe}
          strategy={strategy}
          lotSize={lotSize}
          fastMode={fastMode}
          maxBars={maxBars}
          maxTokens={maxTokens}
          modelName={modelName}
          onAnalysisStart={() => setIsAnalyzing(true)}
          onAnalysisComplete={handleAnalysisComplete}
          selectedSignal={selectedSignal}
        />
        <Journal />
      </div>
    </div>
  );
}
