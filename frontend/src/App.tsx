import { useEffect, useRef, useState } from 'react';
import './styles/global.css';
import './styles/chat.css';

import type { AIOutputHandle } from './components/AIOutput';
import type { AnalysisResult } from './types/analysis';
import { getAgentStatus, setAgentConfig, type AgentStatus } from './services/api';
import { AgentSignal } from './types';
import Header from './components/Header';
import Chart from './components/Chart';
import SidePanel from './components/SidePanel';
import Journal from './components/Journal';
import AIOutput from './components/AIOutput';
import Chat from './components/Chat';
import AgentSettings from './components/AgentSettings';
import SymbolSelector from './components/SymbolSelector';

function App() {
  const [symbol, setSymbol] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('M5');
  const [strategy, setStrategy] = useState('smc');
  const [lotSize, setLotSize] = useState(0.1);
  const [fastMode, setFastMode] = useState(true);
  const [maxBars, setMaxBars] = useState(250);
  const [maxTokens, setMaxTokens] = useState(256);
  const [modelName, setModelName] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAgentSettingsOpen, setIsAgentSettingsOpen] = useState(false);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [selectedSignal, setSelectedSignal] = useState<AgentSignal | null>(null);

  const aiOutputRef = useRef<AIOutputHandle>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      getAgentStatus().then(setAgentStatus).catch(() => setAgentStatus(null));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleRunAnalysis = () => {
    setSelectedSignal(null);
    aiOutputRef.current?.runAnalysis();
  };

  const handleCancelAnalysis = () => {
    aiOutputRef.current?.cancelAnalysis();
  };

  const handlePlaceTrade = () => {
    aiOutputRef.current?.placeTrade();
  };

  const handleAnalysisComplete = (result: AnalysisResult | null) => {
    setAnalysis(result);
    setIsAnalyzing(false);
  };

  const handleToggleAgent = async () => {
    if (!agentStatus) return;
    try {
      const newConfig = { ...agentStatus, enabled: !agentStatus.enabled };
      await setAgentConfig(newConfig);
      setAgentStatus(newConfig);
    } catch (error) {
      console.error('Failed to toggle agent', error);
    }
  };

  const handleSignalSelect = (signal: AgentSignal) => {
    if (signal.symbol) {
      setSymbol(signal.symbol);
    }
    if (signal.timeframe) {
      setTimeframe(signal.timeframe);
    }
    setSelectedSignal(signal);
  };

  return (
    <>
      <Header
        strategy={strategy}
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
        onOpenAgentSettings={() => setIsAgentSettingsOpen(true)}
        agentStatus={agentStatus}
        onToggleAgent={handleToggleAgent}
      />
      <div className="toolbar">
        <SymbolSelector onSymbolChange={setSymbol} value={symbol} />
        <select title="Timeframe" value={timeframe} onChange={e => setTimeframe(e.target.value)}>
          <option value="M1">M1</option>
          <option value="M5">M5</option>
          <option value="M15">M15</option>
          <option value="H1">H1</option>
          <option value="H4">H4</option>
          <option value="D1">D1</option>
        </select>
      </div>
      <AgentSettings isOpen={isAgentSettingsOpen} onClose={() => setIsAgentSettingsOpen(false)} />
      <div className="grid">
        <div className="box">
          <div id="chartWrap">
            <Chart symbol={symbol} timeframe={timeframe} analysis={analysis} />
          </div>
        </div>
        <div className="sidecol">
          <SidePanel onSignalSelected={handleSignalSelect} />
        </div>
      </div>
      <div className="box" style={{ margin: '0 14px 14px' }}>
        <Journal />
      </div>
      <div className="box" style={{ margin: '0 14px 14px' }}>
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
      </div>
      <Chat />
    </>
  );
}

export default App;

