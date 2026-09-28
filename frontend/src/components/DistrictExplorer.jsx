import React, { useState, useEffect } from 'react';
import { getDistricts } from '../api/districts';
import { getDistrictHistory } from '../api/history';
import { getForecastDistrict } from '../api/forecasts';
import ForecastCard from './ForecastCard';
import { AlertCircle, Search, Info, Activity, Droplets, Thermometer, CloudRain, Calendar } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    if (data.type === 'historical') {
      return (
        <div className="bg-white p-3 border border-slate-200 rounded-lg shadow-lg">
          <p className="text-sm font-bold text-slate-800 mb-1">{label}</p>
          <div className="flex items-center text-sm text-blue-700">
            <span className="font-semibold mr-2">Historical groundwater level:</span>
            <span>{data.groundwater_level_m_bgl?.toFixed(3)} m bgl</span>
          </div>
        </div>
      );
    } else {
      const forecastHorizons = {
        '2025-11-01': '+1 Month',
        '2026-01-01': '+3 Months',
        '2026-04-01': '+6 Months'
      };
      const horizonLabel = forecastHorizons[label] || 'Forecast';
      return (
        <div className="bg-white p-3 border border-slate-200 rounded-lg shadow-lg">
          <p className="text-sm font-bold text-slate-800 mb-1">{label}</p>
          <div className="flex flex-col space-y-1">
            <div className="flex items-center text-sm text-amber-600">
              <span className="font-semibold mr-2">Predicted groundwater level:</span>
              <span>{data.forecast_level?.toFixed(3)} m bgl</span>
            </div>
            <div className="text-xs text-slate-500 font-medium">Horizon: {horizonLabel}</div>
          </div>
        </div>
      );
    }
  }
  return null;
};

const DistrictExplorer = () => {
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('');
  
  const [history, setHistory] = useState([]);
  const [forecasts, setForecasts] = useState({ 1: null, 3: null, 6: null });
  
  const [loadingDistricts, setLoadingDistricts] = useState(true);
  const [loadingData, setLoadingData] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDistricts = async () => {
      try {
        const data = await getDistricts();
        setDistricts(data);
        if (data.length > 0) {
          setSelectedDistrict(data[0]);
        }
      } catch (err) {
        setError("Unable to connect to GeoForecast API.");
      } finally {
        setLoadingDistricts(false);
      }
    };
    fetchDistricts();
  }, []);

  useEffect(() => {
    if (!selectedDistrict) return;
    
    const fetchData = async () => {
      setLoadingData(true);
      setError(null);
      try {
        const [histData, f1, f3, f6] = await Promise.all([
          getDistrictHistory(selectedDistrict),
          getForecastDistrict(selectedDistrict, 1),
          getForecastDistrict(selectedDistrict, 3),
          getForecastDistrict(selectedDistrict, 6)
        ]);
        
        setHistory(histData.records || []);
        setForecasts({ 1: f1, 3: f3, 6: f6 });
      } catch (err) {
        setError(`Unable to load data for ${selectedDistrict}.`);
      } finally {
        setLoadingData(false);
      }
    };
    
    fetchData();
  }, [selectedDistrict]);

  // Prepare chart data
  const chartData = [...history.map(r => ({ ...r, type: 'historical' }))];
  
  if (forecasts[1]) chartData.push({ date: forecasts[1].forecast_date ? forecasts[1].forecast_date.split('T')[0] : '2025-11-01', forecast_level: forecasts[1].predicted_groundwater_level_m_bgl, type: 'forecast' });
  if (forecasts[3]) chartData.push({ date: forecasts[3].forecast_date ? forecasts[3].forecast_date.split('T')[0] : '2026-01-01', forecast_level: forecasts[3].predicted_groundwater_level_m_bgl, type: 'forecast' });
  if (forecasts[6]) chartData.push({ date: forecasts[6].forecast_date ? forecasts[6].forecast_date.split('T')[0] : '2026-04-01', forecast_level: forecasts[6].predicted_groundwater_level_m_bgl, type: 'forecast' });

  // Filter to last 5 years for better visibility
  const recentChartData = chartData.filter(d => {
    const dYear = parseInt(d.date.substring(0, 4));
    return dYear >= 2021;
  });

  const latestHistory = history.length > 0 ? history[history.length - 1] : null;

  if (loadingDistricts) {
    return <div className="flex justify-center py-20"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div></div>;
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Page Header */}
      <div className="bg-white p-8 rounded-xl border border-slate-200 shadow-sm text-center">
        <h1 className="text-3xl font-black text-slate-900 tracking-tight mb-3">District Explorer</h1>
        <p className="text-slate-600 font-medium max-w-2xl mx-auto">Explore historical groundwater observations and model-based forecasts for supported Maharashtra districts.</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl flex items-center shadow-sm">
          <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {!error && (
        <>
          {/* Selector & Summary Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-center">
              <label className="block text-sm font-bold text-slate-800 mb-3 uppercase tracking-wider">Select District</label>
              <div className="relative mb-5 max-w-lg">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-slate-400" />
                </div>
                <select 
                  value={selectedDistrict} 
                  onChange={(e) => setSelectedDistrict(e.target.value)}
                  className="block w-full pl-12 pr-10 py-3 border border-slate-300 rounded-lg text-lg font-bold text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 appearance-none bg-slate-50 cursor-pointer transition-colors hover:bg-slate-100"
                >
                  {districts.map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-wrap items-center gap-4 text-sm">
                <div className="flex items-center text-slate-600 bg-slate-50 px-3 py-1.5 rounded-md border border-slate-100">
                  <span className="font-medium mr-2 text-slate-500">Selected district:</span>
                  <strong className="text-slate-800">{selectedDistrict}</strong>
                </div>
                <div className="flex items-center text-slate-600 bg-slate-50 px-3 py-1.5 rounded-md border border-slate-100">
                  <span className="font-medium mr-2 text-slate-500">Latest observation date:</span>
                  <strong className="text-slate-800">2025-10-01</strong>
                </div>
                <div className="flex items-center text-slate-600 bg-emerald-50 px-3 py-1.5 rounded-md border border-emerald-100">
                  <span className="font-medium mr-2 text-emerald-700">Forecast availability:</span>
                  <strong className="text-emerald-800">Ready</strong>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
              <h3 className="text-sm font-bold text-slate-800 mb-4 border-b border-slate-100 pb-3 uppercase tracking-wider">Data Coverage</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-slate-500">District</span>
                  <span className="font-bold text-slate-800">{selectedDistrict}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-slate-500">Latest observation</span>
                  <span className="font-bold text-slate-800">2025-10-01</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-slate-500">Available observations</span>
                  <span className="font-bold text-slate-800">{history.length}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-slate-500">Forecast horizons</span>
                  <span className="font-bold text-blue-600">+1, +3, +6 months</span>
                </div>
                <div className="flex justify-between items-center text-sm pt-2 border-t border-slate-50">
                  <span className="text-slate-500">Data source type</span>
                  <span className="font-medium text-slate-700">Historical & Model</span>
                </div>
              </div>
            </div>
          </div>

          {/* Current Conditions */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h3 className="text-lg font-bold text-slate-800 mb-5">Current Conditions</h3>
            {loadingData ? (
              <div className="animate-pulse flex space-x-4">
                <div className="h-16 bg-slate-100 rounded flex-1"></div>
                <div className="h-16 bg-slate-100 rounded flex-1"></div>
                <div className="h-16 bg-slate-100 rounded flex-1"></div>
                <div className="h-16 bg-slate-100 rounded flex-1"></div>
              </div>
            ) : latestHistory ? (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    <Calendar className="w-3.5 h-3.5 mr-1.5" /> Observation Date
                  </div>
                  <div className="text-xl font-bold text-slate-800">2025-10-01</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    <Droplets className="w-3.5 h-3.5 mr-1.5 text-blue-500" /> Groundwater Level
                  </div>
                  <div className="flex items-baseline">
                    <span className="text-2xl font-black text-slate-800">{latestHistory.groundwater_level_m_bgl?.toFixed(2)}</span>
                    <span className="text-sm font-bold text-slate-500 ml-1">m bgl</span>
                  </div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    <CloudRain className="w-3.5 h-3.5 mr-1.5 text-blue-600" /> Rainfall
                  </div>
                  <div className="flex items-baseline">
                    <span className="text-2xl font-black text-blue-700">{latestHistory.rainfall_mm?.toFixed(1)}</span>
                    <span className="text-sm font-bold text-slate-500 ml-1">mm</span>
                  </div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    <Thermometer className="w-3.5 h-3.5 mr-1.5 text-amber-500" /> Temperature
                  </div>
                  <div className="flex items-baseline">
                    <span className="text-2xl font-black text-amber-600">{latestHistory.temperature_c?.toFixed(1)}</span>
                    <span className="text-sm font-bold text-slate-500 ml-1">°C</span>
                  </div>
                </div>
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                  <div className="flex items-center text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
                    <Activity className="w-3.5 h-3.5 mr-1.5 text-indigo-500" /> GRACE TWS
                  </div>
                  <div className="flex items-baseline">
                    <span className="text-2xl font-black text-indigo-700">{latestHistory.grace_tws?.toFixed(2)}</span>
                    <span className="text-sm font-bold text-slate-500 ml-1">cm</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500">No historical data available.</p>
            )}
          </div>

          {/* Forecast Section */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h3 className="text-xl font-bold text-slate-800 mb-5">District Forecasts</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <ForecastCard horizon={1} forecast={forecasts[1]} loading={loadingData} />
              <ForecastCard horizon={3} forecast={forecasts[3]} loading={loadingData} />
              <ForecastCard horizon={6} forecast={forecasts[6]} loading={loadingData} />
            </div>
          </div>

          {/* Chart Section */}
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="mb-6">
              <h3 className="text-xl font-bold text-slate-800">Groundwater Level — Historical & Forecast</h3>
              <p className="text-sm text-slate-500 mt-1">Historical observations with model-based forecast demonstration.</p>
            </div>
            
            {loadingData ? (
              <div className="h-[400px] bg-slate-50 animate-pulse rounded-lg flex justify-center items-center">
                <span className="text-slate-400 font-medium">Loading visualization...</span>
              </div>
            ) : (
              <div className="h-[450px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={recentChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 12, fill: '#64748b' }} 
                      tickMargin={12}
                      minTickGap={40}
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis 
                      reversed={true} 
                      domain={['auto', 'auto']}
                      tick={{ fontSize: 12, fill: '#64748b', fontWeight: '500' }} 
                      tickMargin={12}
                      axisLine={{ stroke: '#cbd5e1' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                      label={{ value: 'Groundwater level (m bgl)', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 13, fontWeight: 'bold', offset: 0 }}
                    />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                    <ReferenceLine x="2025-10-01" stroke="#94a3b8" strokeDasharray="4 4" label={{ position: 'top', value: 'Today', fill: '#64748b', fontSize: 12, fontWeight: 'bold' }} />
                    
                    <Line 
                      type="monotone" 
                      dataKey="groundwater_level_m_bgl" 
                      name="Historical" 
                      stroke="#2563eb" 
                      strokeWidth={2.5} 
                      dot={false}
                      activeDot={{ r: 6, fill: '#2563eb', stroke: '#fff', strokeWidth: 2 }} 
                    />
                    
                    <Line 
                      type="monotone" 
                      dataKey="forecast_level" 
                      name="Forecast" 
                      stroke="#f59e0b" 
                      strokeWidth={2.5}
                      strokeDasharray="6 4"
                      dot={{ r: 5, fill: '#f59e0b', strokeWidth: 2, stroke: '#fff' }}
                      activeDot={{ r: 7, fill: '#f59e0b', stroke: '#fff', strokeWidth: 2 }} 
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Scientific Interpretation */}
          <div className="bg-blue-50/50 border border-blue-100 p-5 rounded-xl flex gap-4 text-sm text-blue-900 shadow-sm">
            <Info className="w-5 h-5 flex-shrink-0 text-blue-600 mt-0.5" />
            <div className="space-y-1.5 text-blue-800">
              <strong className="block text-blue-900 text-base mb-2">How to read this chart</strong>
              <p>Groundwater level is reported in <strong>metres below ground level (m bgl)</strong>. A larger value indicates a deeper groundwater level.</p>
              <p>Positive predicted change indicates a deeper predicted groundwater level relative to the latest observation.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default DistrictExplorer;
