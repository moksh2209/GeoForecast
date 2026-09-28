import { useState } from 'react';
import Overview from './components/Overview';
import DistrictExplorer from './components/DistrictExplorer';
import ModelPerformance from './components/ModelPerformance';
import { Database, Map as MapIcon, BarChart3, Activity } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Header */}
      <header className="bg-slate-900 text-white border-b border-slate-800 shadow-sm sticky top-0 z-50">
        <div className="w-full max-w-[1800px] mx-auto px-4 md:px-8 xl:px-12">
          <div className="flex flex-col md:flex-row md:items-center justify-between min-h-[64px] py-4 md:py-0 gap-4 md:gap-0">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-blue-400 flex-shrink-0" />
              <div>
                <h1 className="text-xl font-bold tracking-tight text-white leading-tight">GeoForecast</h1>
                <p className="text-xs text-slate-300 font-medium">GRACE-Based Groundwater Forecasting & Spatial Analytics</p>
              </div>
            </div>
            <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0 pb-1 md:pb-0 no-scrollbar">
              <div className="flex items-center space-x-2 sm:space-x-4 min-w-max">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'overview' ? 'bg-slate-800 text-white' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4" />
                    <span>Overview</span>
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('explorer')}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'explorer' ? 'bg-slate-800 text-white' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <MapIcon className="w-4 h-4" />
                    <span>District Explorer</span>
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('performance')}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'performance' ? 'bg-slate-800 text-white' : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4" />
                    <span>Model Performance</span>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full max-w-[1800px] mx-auto px-4 md:px-8 xl:px-12 py-8">
        {activeTab === 'overview' && <Overview navigateToExplorer={() => setActiveTab('explorer')} />}
        {activeTab === 'explorer' && <DistrictExplorer />}
        {activeTab === 'performance' && <ModelPerformance />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-100 border-t border-slate-200 mt-auto">
        <div className="w-full max-w-[1800px] mx-auto px-4 md:px-8 xl:px-12 py-6 flex flex-col sm:flex-row justify-between items-center gap-4 text-sm text-slate-500">
          <div>
            <p>GeoForecast — Scientific Demonstration Project</p>
          </div>
          <div className="text-center sm:text-left">
            <p>Groundwater level measured in metres below ground level (m bgl)</p>
          </div>
          <div>
            <p>React • Vite • FastAPI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
