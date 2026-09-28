import React, { useState, useEffect } from 'react';
import { getDistricts } from '../api/districts';
import { getModels } from '../api/models';
import { getForecastMaharashtra } from '../api/forecasts';
import MaharashtraMap from './MaharashtraMap';
import ForecastCard from './ForecastCard';
import { Database, Layers, CheckCircle2, AlertCircle, Map as MapIcon, Activity } from 'lucide-react';

const Overview = ({ navigateToExplorer }) => {
  const [districts, setDistricts] = useState([]);
  const [models, setModels] = useState([]);
  const [forecasts, setForecasts] = useState({ 1: null, 3: null, 6: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [distList, modelList, f1, f3, f6] = await Promise.all([
          getDistricts(),
          getModels(),
          getForecastMaharashtra(1),
          getForecastMaharashtra(3),
          getForecastMaharashtra(6)
        ]);
        
        setDistricts(distList);
        setModels(modelList);
        setForecasts({ 1: f1, 3: f3, 6: f6 });
        setError(null);
      } catch (err) {
        console.error("Dashboard error:", err);
        setError("Unable to connect to GeoForecast API.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-8 rounded-xl flex flex-col items-center justify-center">
        <AlertCircle className="w-10 h-10 text-red-50 mb-3" />
        <h3 className="text-lg font-bold">API Connection Error</h3>
        <p className="text-sm mt-1">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 xl:space-y-10 pb-12 w-full">
      {/* Header / Hero Section */}
      <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 justify-between items-start lg:items-center">
        <div className="lg:w-1/3 xl:w-2/5">
          <h2 className="text-2xl md:text-3xl font-extrabold text-slate-800 tracking-tight">Dashboard Overview</h2>
          <p className="text-slate-500 mt-2 text-sm md:text-base leading-relaxed">
            GRACE-based groundwater level forecasting system for Maharashtra. 
            Providing predictive insights for water resource management.
          </p>
        </div>
        
        {/* System Status Section */}
        <div className="lg:w-2/3 xl:w-3/5 w-full bg-white p-5 md:p-6 rounded-xl border border-slate-200 shadow-sm">
          <h2 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center">
            <Database className="w-4 h-4 mr-2" />
            System Status
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
            <div className="flex flex-col">
              <span className="text-xs text-slate-500 mb-1">Latest observation</span>
              <div className="flex items-center text-sm font-semibold text-slate-800">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2" />
                2025-10-01
              </div>
            </div>
            <div className="flex flex-col border-l border-slate-100 pl-4">
              <span className="text-xs text-slate-500 mb-1">Supported districts</span>
              <div className="flex items-center text-sm font-semibold text-slate-800">
                <Layers className="w-4 h-4 text-blue-500 mr-2" />
                {districts.length}
              </div>
            </div>
            <div className="flex flex-col border-l border-slate-100 pl-4">
              <span className="text-xs text-slate-500 mb-1">Forecast horizons</span>
              <div className="flex gap-2">
                <span className="text-sm font-semibold text-slate-800">+1 / +3 / +6 months</span>
              </div>
            </div>
            <div className="flex flex-col border-l border-slate-100 pl-4">
              <span className="text-xs text-slate-500 mb-1">Forecast status</span>
              <div className="text-sm font-semibold text-amber-600">
                Historical-input demonstration
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Maharashtra Forecasts */}
      <div className="bg-white p-6 md:p-8 rounded-xl border border-slate-200 shadow-sm w-full">
        <div className="mb-6 md:mb-8 border-b border-slate-100 pb-4">
          <h2 className="text-xl md:text-2xl font-bold text-slate-800">Maharashtra-wide Forecast</h2>
          <p className="text-sm md:text-base text-slate-500 mt-1">State-level groundwater level predictions from the production forecasting pipeline.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 xl:gap-8 w-full">
          <ForecastCard horizon={1} forecast={forecasts[1]} />
          <ForecastCard horizon={3} forecast={forecasts[3]} />
          <ForecastCard horizon={6} forecast={forecasts[6]} />
        </div>
      </div>
      
      {/* Map + Information Area */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.5fr)_minmax(360px,0.8fr)] gap-6 lg:gap-8 w-full">
        {/* District Forecast Map */}
        <div className="bg-white p-6 md:p-8 rounded-xl border border-slate-200 shadow-sm flex flex-col h-full">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
            <div>
              <h2 className="text-xl md:text-2xl font-bold text-slate-800">District Forecast Map</h2>
              <p className="text-sm md:text-base text-slate-500 mt-1">{districts.length} supported districts</p>
            </div>
            <button 
              onClick={navigateToExplorer}
              className="text-sm font-medium text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 px-4 py-2.5 rounded-lg transition-colors whitespace-nowrap"
            >
              Open Full Explorer
            </button>
          </div>
          
          <div className="flex-grow rounded-xl overflow-hidden border border-slate-200">
            <MaharashtraMap districts={districts} />
          </div>
          
          <div className="mt-5 flex items-start">
            <span className="text-xs text-slate-500 flex items-start bg-slate-50 px-4 py-3 rounded-lg border border-slate-100 w-full">
              <AlertCircle className="w-4 h-4 mr-2 text-blue-500 flex-shrink-0 mt-0.5" />
              <span>Markers represent district forecast locations and do not represent a continuous spatial prediction surface.</span>
            </span>
          </div>
        </div>

        {/* Adjacent Information Area */}
        <div className="bg-white p-6 md:p-8 rounded-xl border border-slate-200 shadow-sm flex flex-col h-full">
          <h2 className="text-xl md:text-2xl font-bold text-slate-800 mb-6 border-b border-slate-100 pb-4">Interpretation Guide</h2>
          
          <div className="space-y-6 flex-grow flex flex-col">
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-100">
              <h3 className="text-sm font-bold text-slate-800 mb-3 flex items-center">
                <MapIcon className="w-4 h-4 mr-2 text-slate-500" />
                Map Usage
              </h3>
              <p className="text-sm text-slate-600 mb-2">Click on any district marker to view localized predictions. The popup displays:</p>
              <ul className="text-sm text-slate-600 space-y-1.5 list-disc pl-5">
                <li>Current observed level</li>
                <li>+1 month prediction</li>
                <li>+3 months prediction</li>
                <li>+6 months prediction</li>
              </ul>
            </div>

            <div className="bg-blue-50/50 rounded-xl p-5 border border-blue-100">
              <h3 className="text-sm font-bold text-blue-900 mb-3 flex items-center">
                <Activity className="w-4 h-4 mr-2 text-blue-600" />
                Scientific Context
              </h3>
              <p className="text-sm text-blue-800 leading-relaxed">
                Groundwater levels are measured in <span className="font-semibold">metres below ground level (m bgl)</span>. 
                Higher m bgl values indicate a deeper water table (reduced availability). Lower values indicate a shallower table.
              </p>
            </div>

            <div className="bg-emerald-50/50 rounded-xl p-5 border border-emerald-100 mt-auto">
              <h3 className="text-sm font-bold text-emerald-900 mb-3 flex items-center">
                <CheckCircle2 className="w-4 h-4 mr-2 text-emerald-600" />
                Model Confidence
              </h3>
              <p className="text-sm text-emerald-800 leading-relaxed">
                Forecasts are generated using a machine learning pipeline trained on GRACE satellite data, climatic indicators (ENSO), and historical groundwater observations. Test R² and RMSE metrics are available in the performance tab.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Forecasting Framework Methodology */}
      <div className="bg-white p-6 md:p-8 rounded-xl border border-slate-200 shadow-sm w-full">
        <h2 className="text-xl md:text-2xl font-bold text-slate-800 mb-6 text-center">Forecasting Framework</h2>
        <div className="flex flex-col md:flex-row items-center justify-center gap-3 md:gap-4 lg:gap-6 text-sm text-slate-600">
          <div className="bg-slate-50 px-6 py-4 rounded-xl border border-slate-200 font-medium w-full md:w-auto text-center shadow-sm">
            GRACE Satellite
          </div>
          <div className="text-blue-400 rotate-90 md:rotate-0 font-bold text-xl">→</div>
          <div className="bg-slate-50 px-6 py-4 rounded-xl border border-slate-200 font-medium w-full md:w-auto text-center shadow-sm">
            Climate & ENSO
          </div>
          <div className="text-blue-400 rotate-90 md:rotate-0 font-bold text-xl">→</div>
          <div className="bg-slate-50 px-6 py-4 rounded-xl border border-slate-200 font-medium w-full md:w-auto text-center shadow-sm">
            Groundwater Obs.
          </div>
          <div className="text-blue-400 rotate-90 md:rotate-0 font-bold text-xl">→</div>
          <div className="bg-blue-50 text-blue-700 px-6 py-4 rounded-xl border border-blue-200 font-bold w-full md:w-auto text-center shadow-sm">
            ML Forecasting
          </div>
          <div className="text-blue-400 rotate-90 md:rotate-0 font-bold text-xl">→</div>
          <div className="bg-emerald-50 text-emerald-700 px-6 py-4 rounded-xl border border-emerald-200 font-bold w-full md:w-auto text-center shadow-sm">
            Dashboard
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;
