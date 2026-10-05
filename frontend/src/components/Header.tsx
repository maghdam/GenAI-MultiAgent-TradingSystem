import AppNav from './AppNav';
import type { V2CTraderAccount } from '../services/api';

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
  ctraderAccounts?: V2CTraderAccount[];
  ctraderAccountBusy?: boolean;
  ctraderAccountSwitchInProgress?: boolean;
  liveTradingArmed?: boolean;
  onCTraderAccountChange?: (accountId: number) => void;
  /* new props for the toolbar */
  symbol?: string;
  timeframe?: string;
  onTimeframeChange?: (value: string) => void;
}

const TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'];

const DEFAULT_FEED_STATUS: StatusChip = { status: 'wait', label: 'Feed' };
const DEFAULT_LLM_STATUS: StatusChip = { status: 'wait', label: 'Model' };


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
  ctraderAccounts = [],
  ctraderAccountBusy = false,
  ctraderAccountSwitchInProgress = false,
  liveTradingArmed = false,
  onCTraderAccountChange,
  timeframe,
  onTimeframeChange,
}: HeaderProps) {
  const resolvedStrategyOptions = Array.from(new Set([...(strategyOptions || []), strategy]));
  const selectedCTraderAccount = ctraderAccounts.find((account) => account.selected);
  const activeCTraderAccount = ctraderAccounts.find((account) => account.active);
  const accountLabel = (account: V2CTraderAccount) => {
    const broker = account.broker_title || 'cTrader';
    const login = account.trader_login ?? account.account_id;
    return `${broker} · ${account.account_type === 'live' ? 'Live' : 'Demo'} · ${login}`;
  };

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
      <AppNav right={(
        <>
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
            title="Trade setup"
          >
            ⚙
          </button>
        </>
      )} />

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

        {onCTraderAccountChange && (
          <>
            <div className="ta-toolbar__divider" />
            <div className="ta-toolbar__group">
              <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--ta-text-secondary)' }}>
                Account
                <select
                  className="ta-select ta-select--sm"
                  aria-label="cTrader account"
                  title="Accounts granted to the current cTrader Open API access token"
                  value={selectedCTraderAccount?.account_id ?? ''}
                  onChange={(event) => onCTraderAccountChange(Number(event.target.value))}
                  disabled={ctraderAccountBusy || ctraderAccountSwitchInProgress || ctraderAccounts.length === 0}
                  style={{ minWidth: '210px' }}
                >
                  {!selectedCTraderAccount && <option value="">Select cTrader account</option>}
                  {ctraderAccounts.map((account) => (
                    <option key={account.account_id} value={account.account_id}>
                      {accountLabel(account)}{account.active ? ' · Active' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <span className="ta-status" title="Currently authenticated cTrader transport account.">
                <span className={`ta-status__dot ${activeCTraderAccount ? 'ta-status__dot--ok' : 'ta-status__dot--wait'}`} />
                {activeCTraderAccount ? `Active: ${accountLabel(activeCTraderAccount)}` : 'Active: none'}
              </span>
              {selectedCTraderAccount?.account_type === 'live' && (
                <span
                  className="ta-status"
                  title={
                    liveTradingArmed
                      ? 'Selected Live account is armed for real-money entries; all normal execution gates still apply.'
                      : 'Selected Live account is disarmed; new real-money entries are blocked until Live Trading is armed in System.'
                  }
                >
                  <span className={`ta-status__dot ${liveTradingArmed ? 'ta-status__dot--bad' : 'ta-status__dot--wait'}`} />
                  {liveTradingArmed ? 'LIVE MONEY · ARMED' : 'LIVE MONEY · DISARMED'}
                </span>
              )}
              {ctraderAccountSwitchInProgress && (
                <span className="ta-status">
                  <span className="ta-status__dot ta-status__dot--wait" />
                  Switching account…
                </span>
              )}
            </div>
          </>
        )}

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
