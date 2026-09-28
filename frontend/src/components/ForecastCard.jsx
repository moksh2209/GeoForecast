import React from 'react';
import { Calendar, BrainCircuit, Droplets } from 'lucide-react';

const ForecastCard = ({ horizon, forecast, loading, error }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm animate-pulse h-48">
        <div className="h-4 bg-slate-200 rounded w-1/3 mb-4"></div>
        <div className="h-8 bg-slate-200 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-slate-200 rounded w-1/4"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 rounded-xl border border-red-100 p-6 shadow-sm h-48 flex flex-col justify-center">
        <p className="text-red-500 font-medium text-sm text-center">Unable to load forecast</p>
        <p className="text-red-400 text-xs text-center mt-1">{error}</p>
      </div>
    );
  }

  if (!forecast) return null;

  const isPositive = forecast.predicted_change_m_bgl > 0;
  
  // Directly use the API provided date
  const forecastDate = forecast.forecast_date ? forecast.forecast_date.split('T')[0] : 'N/A';
  const inputDate = '2025-10-01'; // Given by system status
  
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow h-full flex flex-col relative overflow-hidden group">
      {/* Top Banner */}
      <div className="bg-slate-50 border-b border-slate-100 px-5 py-3 flex justify-between items-center group-hover:bg-blue-50/50 transition-colors">
        <div className="font-bold text-slate-800 tracking-wide">+{horizon} MONTH{horizon > 1 ? 'S' : ''}</div>
        <div className="flex items-center text-xs font-bold text-slate-600 bg-white px-2 py-1 rounded border border-slate-200 shadow-sm group-hover:border-blue-200 group-hover:text-blue-700 transition-colors">
          <Calendar className="w-3 h-3 mr-1" />
          {forecastDate}
        </div>
      </div>
      
      {/* Main Content */}
      <div className="p-6 flex-grow flex flex-col justify-center items-center text-center">
        <div className="flex items-baseline justify-center mb-1">
          <span className="text-4xl font-black text-slate-800 tracking-tight">
            {forecast.predicted_groundwater_level_m_bgl.toFixed(3)}
          </span>
          <span className="text-base font-bold text-slate-500 ml-1.5">m bgl</span>
        </div>
        
        <div className="flex flex-col items-center mt-3 bg-slate-50 px-4 py-2 rounded-lg border border-slate-100 w-full">
          <div className={`text-sm font-semibold flex items-center ${isPositive ? 'text-amber-600' : 'text-blue-600'}`}>
            <span>Change: {isPositive ? '+' : ''}{forecast.predicted_change_m_bgl.toFixed(3)} m bgl</span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {isPositive ? "Deeper predicted groundwater level" : "Shallower predicted groundwater level"}
          </p>
        </div>
      </div>
      
      {/* Footer Info */}
      <div className="bg-slate-50 px-5 py-3.5 text-xs text-slate-500 border-t border-slate-100 flex flex-col space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <BrainCircuit className="w-4 h-4 mr-1.5 text-blue-500" />
            <span>Model: <span className="font-semibold text-slate-700">{forecast.model}</span></span>
          </div>
        </div>
        <div className="flex items-center justify-between border-t border-slate-200/60 pt-2">
          <div className="text-[11px] text-slate-400">
            Latest input: <span className="font-medium text-slate-500">{inputDate}</span>
          </div>
          <div className="text-amber-600 font-medium italic text-[10px] uppercase tracking-wider">
            Historical-input
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastCard;
