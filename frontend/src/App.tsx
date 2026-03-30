import { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import './styles/global.css';

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const StrategyStudioPage = lazy(() => import('./pages/StrategyStudio/index'));
const StrategyStudioResultsPage = lazy(() => import('./pages/StrategyStudio/Results'));
const HeavyweightChecklistPage = lazy(() => import('./pages/HeavyweightChecklist'));
const WorkbenchPage = lazy(() => import('./pages/Workbench'));

export default function App() {
  return (
    <Suspense fallback={<div className="v2-banner">Loading workspace...</div>}>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/strategy-studio" element={<StrategyStudioPage />} />
        <Route path="/strategy-studio/results" element={<StrategyStudioResultsPage />} />
        <Route path="/heavyweight-checklist" element={<HeavyweightChecklistPage />} />
        <Route path="/workbench" element={<WorkbenchPage />} />
        <Route path="/v2" element={<WorkbenchPage />} />
      </Routes>
    </Suspense>
  );
}
