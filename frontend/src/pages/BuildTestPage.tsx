import AppNav from '../components/AppNav';
import ResearchValidationPanel from '../components/ResearchValidationPanel';
import StrategyStudioPage from './StrategyStudio';

export default function BuildTestPage() {
  return (
    <div className="ta-app">
      <AppNav right={<span className="ta-status"><span className="ta-status__dot ta-status__dot--wait" />Research workspace</span>} />
      <main className="v2-shell">
        <section className="v2-hero" style={{ marginBottom: 16 }}>
          <div>
            <p className="v2-kicker">Build & Test</p>
            <h1>Turn an idea into explicit, testable strategy rules.</h1>
            <p className="v2-lead">Idea → rules → backtest → validation → paper evidence → deployment eligibility. GenAI assists research; it cannot approve or execute trades.</p>
          </div>
        </section>
        <StrategyStudioPage />
        <ResearchValidationPanel />
      </main>
    </div>
  );
}
