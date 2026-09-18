import AppNav from '../components/AppNav';
import StrategyStudioResultsPage from './StrategyStudio/Results';

export default function BuildTestResultsPage() {
  return (
    <div className="ta-app">
      <AppNav right={<span className="ta-status"><span className="ta-status__dot ta-status__dot--wait" />Research results</span>} />
      <StrategyStudioResultsPage />
    </div>
  );
}
