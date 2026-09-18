import { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import './styles/global.css';

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const BuildTestPage = lazy(() => import('./pages/BuildTestPage'));
const BuildTestResultsPage = lazy(() => import('./pages/BuildTestResultsPage'));
const SystemPage = lazy(() => import('./pages/SystemPage'));

export default function App() {
  return (
    <Suspense fallback={<div className="v2-banner">Loading workspace...</div>}>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/build-test" element={<BuildTestPage />} />
        <Route path="/build-test/results" element={<BuildTestResultsPage />} />
        <Route path="/system" element={<SystemPage />} />
      </Routes>
    </Suspense>
  );
}
