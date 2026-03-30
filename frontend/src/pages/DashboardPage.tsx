import { useEffect, useRef, useState } from 'react';

import type { AIOutputHandle } from '../components/AIOutput';
import type { AnalysisResult } from '../types/analysis';
import {
  getV2Strategies,
  getV2Status,
  setV2Config,
  startV2Engine,
  stopV2Engine,
  type V2StrategyInfo,
  type V2Status,
  type V2WatchlistItem,
} from '../services/api';
import type { AgentSignal } from '../types';
import Header, { type DashboardEngineStatus } from '../components/Header';
import AgentSettings from '../components/AgentSettings';
import Chart from '../components/Chart';
import SidePanel from '../components/SidePanel';
import Journal from '../components/Journal';
import AIOutput from '../components/AIOutput';
import SymbolSelector from '../components/SymbolSelector';

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
  const [selectedSignal, setSelectedSignal] = useState<AgentSignal | null>(null);
  const [v2Strategies, setV2Strategies] = useState<V2StrategyInfo[]>([]);

  const aiOutputRef = useRef<AIOutputHandle>(null);

  const loadDashboardState = async () => {
    const [strategies, nextStatus] = await Promise.all([getV2Strategies(), getV2Status()]);
    setV2Strategies(strategies);
    setStatus(nextStatus);
    setLotSize((current) => (current === 0.01 ? nextStatus.config.paper_trade_size || current : current));
    if (!strategies.some((item) => item.key === strategy) && strategies[0]?.key) {
      setStrategy(strategies[0].key);
    }
  };

  useEffect(() => {
    loadDashboardState().catch(() => { setV2Strategies([]); setStatus(null); });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      getV2Status().then(setStatus).catch(() => setStatus(null));
    }, 6000);
    return () => clearInterval(interval);
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

  const handleRunAnalysis = () => { setSelectedSignal(null); aiOutputRef.current?.runAnalysis(); };
  const handleCancelAnalysis = () => { aiOutputRef.current?.cancelAnalysis(); };
  const handlePlaceTrade = () => { aiOutputRef.current?.placeTrade(); };
  const handleAnalysisComplete = (result: AnalysisResult | null) => { setAnalysis(result); setIsAnalyzing(false); };

  const handleToggleAgent = async () => {
    try {
      if (status?.config.enabled) { await stopV2Engine(); } else { await startV2Engine(); }
      await loadDashboardState();
    } catch (error) { console.error('Failed to toggle engine', error); }
  };

  const handleWatchCurrent = async () => {
    if (!status) return;
    try {
      const nextItem: V2WatchlistItem = { symbol, timeframe, strategy, enabled: true, params: {} };
      const existing = status.config.watchlist || [];
      const deduped = existing.filter((item) => !(item.symbol === symbol && item.timeframe === timeframe));
      await setV2Config({ ...status.config, paper_trade_size: lotSize, watchlist: [...deduped, nextItem] });
      await loadDashboardState();
    } catch (error) { console.error('Failed to add to watchlist', error); }
  };

  const handleReloadStrategies = async () => {
    try { await loadDashboardState(); }
    catch (error) { console.error('Failed to refresh', error); }
  };

  const handleSignalSelect = (signal: AgentSignal) => {
    if (signal.symbol) setSymbol(signal.symbol);
    if (signal.timeframe) setTimeframe(signal.timeframe);
    setSelectedSignal(signal);
  };

  return (
    <div className="ta-app">
      <Header
        strategy={strategy}
        strategyOptions={strategyOptions}
        onStrategyChange={setStrategy}
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
        feedStatus={feedStatus}
        llmStatus={llmStatus}
        onOpenSettings={() => setIsAgentSettingsOpen(true)}
        engineStatus={engineStatus}
        onToggleEngine={handleToggleAgent}
        onWatchCurrent={handleWatchCurrent}
        onRefreshStrategies={handleReloadStrategies}
        symbol={symbol}
        timeframe={timeframe}
        onTimeframeChange={setTimeframe}
      />

      {/* Symbol selector row — compact */}
      <div className="ta-toolbar" style={{ paddingTop: '6px', paddingBottom: '6px', borderBottom: 'none' }}>
        <SymbolSelector onSymbolChange={setSymbol} value={symbol} />
      </div>

      <AgentSettings isOpen={isAgentSettingsOpen} onClose={() => setIsAgentSettingsOpen(false)} />

      {/* ─── Main Layout: Chart + Sidebar ─── */}
      <div className="ta-main">
        <div className="ta-main__chart">
          <Chart symbol={symbol} timeframe={timeframe} analysis={analysis} />
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
