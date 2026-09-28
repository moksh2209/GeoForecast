import React, { useState, useEffect } from 'react';
import { getModels } from '../api/models';
import { AlertCircle, Beaker, Layers, Calendar, BarChart3, ShieldCheck, Info } from 'lucide-react';

const ModelPerformance = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const data = await getModels();
        setModels(data);
        setError(null);
      } catch (err) {
        setError("Unable to connect to GeoForecast API.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchModels();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Page Header */}
      <div className="bg-white p-8 rounded-xl border border-slate-200 shadow-sm text-center">
        <h1 className="text-3xl font-black text-slate-900 tracking-tight mb-3">Model Performance</h1>
        <p className="text-slate-600 font-medium max-w-2xl mx-auto">Evaluation of production forecasting models on held-out chronological test sets.</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl flex items-center shadow-sm">
          <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
          <p className="text-sm font-medium">{error}</p>
        </div>
      )}

      {!error && (
        <>
          {/* Key Metric Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
              <Layers className="w-5 h-5 text-blue-500 mb-2" />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Production Models</span>
              <span className="text-2xl font-black text-slate-800">6</span>
            </div>
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
              <Calendar className="w-5 h-5 text-indigo-500 mb-2" />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Forecast Horizons</span>
              <span className="text-2xl font-black text-slate-800">3</span>
            </div>
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
              <BarChart3 className="w-5 h-5 text-emerald-500 mb-2" />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Evaluation Metrics</span>
              <span className="text-xl font-bold text-slate-800">R² + RMSE</span>
            </div>
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center text-center">
              <ShieldCheck className="w-5 h-5 text-amber-500 mb-2" />
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Evaluation Type</span>
              <span className="text-sm font-bold text-slate-800">Held-out test data</span>
            </div>
          </div>

          {/* Evaluation Scope & Explanations */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 shadow-sm">
              <h4 className="text-sm font-bold text-slate-800 mb-3 uppercase tracking-wider">Evaluation Scope</h4>
              <div className="space-y-3">
                <div>
                  <span className="inline-block px-2 py-0.5 bg-slate-200 text-slate-700 text-[10px] font-bold rounded mb-1">Experiment 5</span>
                  <p className="text-xs text-slate-600 font-medium">Maharashtra-level production model selection.</p>
                </div>
                <div>
                  <span className="inline-block px-2 py-0.5 bg-slate-200 text-slate-700 text-[10px] font-bold rounded mb-1">Experiment 6</span>
                  <p className="text-xs text-slate-600 font-medium">District-level spatial modeling baseline.</p>
                </div>
              </div>
            </div>
            
            <div className="bg-blue-50 p-6 rounded-xl border border-blue-100 shadow-sm flex flex-col justify-center">
              <h4 className="text-sm font-bold text-blue-900 mb-3 uppercase tracking-wider">R² Explanation</h4>
              <p className="text-sm text-blue-800 leading-relaxed font-medium">
                R² measures the proportion of variance explained by the model on the held-out test set. 
                <strong className="block mt-2">It is not classification accuracy.</strong>
              </p>
            </div>
            
            <div className="bg-amber-50 p-6 rounded-xl border border-amber-100 shadow-sm flex flex-col justify-center">
              <h4 className="text-sm font-bold text-amber-900 mb-3 uppercase tracking-wider">RMSE Explanation</h4>
              <p className="text-sm text-amber-800 font-medium mb-2">
                RMSE represents the root mean squared prediction error.
              </p>
              <div className="flex items-center text-sm text-amber-900 font-bold mb-2">
                <span className="bg-amber-200/50 px-2 py-0.5 rounded">Unit: m bgl</span>
              </div>
              <p className="text-xs text-amber-700 font-medium">Lower RMSE indicates smaller prediction error.</p>
            </div>
          </div>

          {/* Important Comparison Note */}
          <div className="flex justify-center">
            <div className="inline-flex items-center bg-white px-5 py-3 rounded-lg border border-slate-200 shadow-sm max-w-3xl text-center">
              <Info className="w-5 h-5 text-slate-400 mr-3 flex-shrink-0" />
              <p className="text-xs font-medium text-slate-600">
                Maharashtra-level and district-level results use different evaluation scopes and should not be interpreted as directly equivalent model comparisons.
              </p>
            </div>
          </div>

          {/* Model Results Table */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex items-center">
              <h3 className="text-lg font-bold text-slate-800">Model Results</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-sm">
                <thead className="bg-white">
                  <tr>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Level</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Horizon</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Model Type</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Test R²</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Test RMSE</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Experiment</th>
                    <th scope="col" className="px-6 py-4 text-left font-bold text-slate-500 uppercase tracking-wider text-xs">Test Samples</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-100">
                  {models.map((model, idx) => {
                    const isMah = model.geographic_level.toLowerCase() === 'maharashtra';
                    return (
                      <tr key={idx} className="hover:bg-slate-50/80 transition-colors group">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-bold ${isMah ? 'bg-indigo-50 text-indigo-700 border border-indigo-100' : 'bg-emerald-50 text-emerald-700 border border-emerald-100'}`}>
                            {model.geographic_level}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-slate-700 font-semibold">
                          +{model.horizon_months} Months
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-slate-600 font-medium">
                            {model.model_type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-blue-700 font-black tabular-nums tracking-tight text-base">
                            {model.test_r2?.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-amber-700 font-black tabular-nums tracking-tight text-base">
                            {model.test_rmse?.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-slate-500 font-medium text-xs">
                          Exp {model.experiment}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-slate-500 font-medium tabular-nums">
                          {model.test_samples ? model.test_samples : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelPerformance;
