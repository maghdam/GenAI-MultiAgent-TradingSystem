import { useState } from 'react';

import AppNav from '../components/AppNav';
import ResearchValidationPanel from '../components/ResearchValidationPanel';
import StrategyStudioPage from './StrategyStudio';

export default function BuildTestPage() {
  const [showAdvancedEvidence, setShowAdvancedEvidence] = useState(false);

  return (
    <div className="ta-app">
      <AppNav right={<span className="ta-status"><span className="ta-status__dot ta-status__dot--wait" />Research workspace</span>} />

      <main className="build-shell">
        <header className="build-heading">
          <div>
            <p className="build-kicker">Build & Test</p>
            <h1>Strategy research workspace</h1>
            <p>Define the idea, draft explicit rules, backtest them, and earn lifecycle evidence before anything can progress toward deployment.</p>
          </div>
          <button
            className="ta-btn ta-btn--ghost"
            type="button"
            onClick={() => setShowAdvancedEvidence((value) => !value)}
          >
            {showAdvancedEvidence ? 'Hide advanced evidence' : 'Advanced evidence'}
          </button>
        </header>

        <StrategyStudioPage />

        {showAdvancedEvidence && (
          <div className="build-advanced-evidence">
            <ResearchValidationPanel />
          </div>
        )}
      </main>
    </div>
  );
}
