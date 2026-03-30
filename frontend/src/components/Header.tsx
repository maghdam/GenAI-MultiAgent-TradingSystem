import { Link, useLocation } from 'react-router-dom';

interface StatusChip {
  status: 'ok' | 'bad' | 'wait' | 'warn';
  label: string;
}

export interface DashboardEngineStatus {
  enabled: boolean;
  running: boolean;
  loopActive: boolean;
  watchlistCount: number;
  mode: string;
}

interface HeaderProps {
  strategy: string;
  strategyOptions?: string[];
  onStrategyChange: (value: string) => void;
  lotSize: number;
  onLotSizeChange: (value: number) => void;
  fastMode: boolean;
  onFastModeChange: (value: boolean) => void;
  maxBars: number;
  onMaxBarsChange: (value: number) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
  modelName: string;
  onModelNameChange: (value: string) => void;
  isAnalyzing: boolean;
  onRunAnalysis: () => void;
  onCancelAnalysis: () => void;
  onPlaceTrade: () => void;
  feedStatus?: StatusChip;
  llmStatus?: StatusChip;
  engineStatus: DashboardEngineStatus | null;
  onWatchCurrent?: () => void;
  onToggleEngine?: () => void;
  onOpenSettings?: () => void;
  onRefreshStrategies?: () => void;
  /* new props for the toolbar */
  symbol?: string;
  timeframe?: string;
  onTimeframeChange?: (value: string) => void;
}

const TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'];

const DEFAULT_FEED_STATUS: StatusChip = { status: 'wait', label: 'Feed' };
const DEFAULT_LLM_STATUS: StatusChip = { status: 'wait', label: 'Model' };

const NAV_LINKS = [
  { to: '/', label: 'Dashboard' },
  { to: '/strategy-studio', label: 'Strategy Studio' },
  { to: '/workbench', label: 'Workbench' },
  { to: '/heavyweight-checklist', label: 'Checklist' },
];

export default function Header({
  strategy,
  strategyOptions,
  onStrategyChange,
  lotSize,
  onLotSizeChange,
  isAnalyzing,
  onRunAnalysis,
  onCancelAnalysis,
  onPlaceTrade,
  feedStatus = DEFAULT_FEED_STATUS,
  llmStatus = DEFAULT_LLM_STATUS,
  engineStatus,
  onWatchCurrent,
  onToggleEngine,
  onOpenSettings,
  onRefreshStrategies,
  timeframe,
  onTimeframeChange,
}: HeaderProps) {
  const location = useLocation();
  const resolvedStrategyOptions = Array.from(new Set([...(strategyOptions || []), strategy]));

  const engineLabel = engineStatus
    ? engineStatus.enabled
      ? engineStatus.loopActive ? 'Scanning' : 'Idle'
      : 'Off'
    : '…';

  const engineDotClass = engineStatus
    ? engineStatus.enabled
      ? engineStatus.loopActive ? 'ta-status__dot--ok' : 'ta-status__dot--wait'
      : 'ta-status__dot--bad'
    : 'ta-status__dot--wait';

  return (
    <>
      {/* ─── Navbar ─── */}
      <nav className="ta-navbar">
        <div className="ta-navbar__brand">
          <div className="ta-navbar__brand-icon">TA</div>
          <span>TradeAgent</span>
        </div>

        <div className="ta-navbar__nav">
          {NAV_LINKS.map((link) => (
            link.to === '/' ? (
              <Link
                key={link.to}
                to={link.to}
                className={`ta-navbar__link${location.pathname === link.to ? ' ta-navbar__link--active' : ''}`}
              >
                {link.label}
              </Link>
            ) : (
              <a
                key={link.to}
                href={link.to}
                target="_blank"
                rel="noopener noreferrer"
                className="ta-navbar__link"
              >
                {link.label}
              </a>
            )
          ))}
        </div>

        <div className="ta-navbar__spacer" />

        <div className="ta-navbar__status">
          <span className="ta-status">
            <span className={`ta-status__dot ta-status__dot--${feedStatus.status}`} />
            {feedStatus.label}
          </span>
          <span className="ta-status">
            <span className={`ta-status__dot ta-status__dot--${llmStatus.status}`} />
            {llmStatus.label}
          </span>
          <span className="ta-status">
            <span className={`ta-status__dot ${engineDotClass}`} />
            Engine: {engineLabel}
          </span>

          <button
            className="ta-btn ta-btn--ghost ta-btn--sm"
            type="button"
            onClick={onOpenSettings}
            title="Control Panel"
          >
            ⚙
          </button>
        </div>
      </nav>

      {/* ─── Toolbar ─── */}
      <div className="ta-toolbar">
        <div className="ta-toolbar__group">
          <select
            className="ta-select ta-select--sm"
            title="Strategy"
            value={strategy}
            onChange={(e) => onStrategyChange(e.target.value)}
          >
            {resolvedStrategyOptions.map((name) => (
              <option key={name} value={name}>
                {name.replace(/_/g, ' ').toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {/* Timeframe pills */}
        {onTimeframeChange && (
          <>
            <div className="ta-toolbar__divider" />
            <div className="ta-tf-group">
              {TIMEFRAMES.map((tf) => (
                <button
                  key={tf}
                  type="button"
                  className={`ta-tf-btn${timeframe === tf ? ' ta-tf-btn--active' : ''}`}
                  onClick={() => onTimeframeChange(tf)}
                >
                  {tf}
                </button>
              ))}
            </div>
          </>
        )}

        <div className="ta-toolbar__divider" />

        <div className="ta-toolbar__group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--ta-text-secondary)' }}>
            Size
            <input
              className="ta-input ta-input--sm ta-input--mono"
              type="number"
              min="0.01"
              step="0.01"
              value={lotSize}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                onLotSizeChange(Number.isNaN(v) ? 0.01 : Math.max(0.01, v));
              }}
              style={{ width: '72px' }}
            />
          </label>
        </div>

        <div className="ta-toolbar__spacer" />

        {/* Action buttons */}
        <div className="ta-toolbar__group">
          <button className="ta-btn ta-btn--sm" type="button" onClick={onWatchCurrent}>
            + Watch
          </button>
          <button className="ta-btn ta-btn--sm" type="button" onClick={onToggleEngine} disabled={!engineStatus}>
            {engineStatus?.enabled ? '⏹ Stop' : '▶ Start'}
          </button>
          <button className="ta-btn ta-btn--sm" type="button" onClick={onRefreshStrategies}>
            ↻ Reload
          </button>
        </div>

        <div className="ta-toolbar__divider" />

        <div className="ta-toolbar__group">
          {isAnalyzing && (
            <button className="ta-btn ta-btn--danger ta-btn--sm" type="button" onClick={onCancelAnalysis}>
              ✕ Cancel
            </button>
          )}
          <button className="ta-btn ta-btn--primary ta-btn--sm" type="button" onClick={onRunAnalysis} disabled={isAnalyzing}>
            {isAnalyzing ? '⟳ Analyzing…' : '⚡ Analyze'}
          </button>
          <button className="ta-btn ta-btn--success ta-btn--sm" type="button" onClick={onPlaceTrade} disabled={isAnalyzing}>
            ↗ Place Trade
          </button>
        </div>
      </div>
    </>
  );
}
