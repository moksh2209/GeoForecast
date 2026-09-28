import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { getForecastDistrict } from '../api/forecasts';
import { getDistrictHistory } from '../api/history';

// Approx center of Maharashtra
const center = [19.7515, 75.7139];

// Hardcoded approx coordinates for demonstration since backend doesn't serve them
const districtCoords = {
  Pune: [18.5204, 73.8567],
  Nashik: [19.9975, 73.7898],
  Nagpur: [21.1458, 79.0882],
  Ahmednagar: [19.0952, 74.7496],
  Akola: [20.7059, 77.0082],
  Amravati: [20.9320, 77.7523],
  Aurangabad: [19.8762, 75.3433],
  Beed: [18.9894, 75.7601],
  Bhandara: [21.1691, 79.6644],
  Buldana: [20.5312, 76.1805],
  Chandrapur: [19.9615, 79.2961],
  Dhule: [20.9042, 74.7749],
  Gadchiroli: [20.1812, 79.9918],
  Gondia: [21.4624, 80.1960],
  Hingoli: [19.7153, 77.1472],
  Jalgaon: [21.0077, 75.5626],
  Jalna: [19.8297, 75.8800],
  Kolhapur: [16.7050, 74.2433],
  Latur: [18.4088, 76.5604],
  Nanded: [19.1383, 77.3210],
  Nandurbar: [21.3736, 74.2443],
  Osmanabad: [18.1856, 76.0419],
  Parbhani: [19.2668, 76.7748],
  Raigad: [18.5158, 72.8712],
  Ratnagiri: [16.9902, 73.3120],
  Sangli: [16.8524, 74.5815],
  Satara: [17.6914, 74.0009],
  Sindhudurg: [16.1666, 73.7431],
  Solapur: [17.6599, 75.9064],
  Thane: [19.2183, 72.9781],
  Wardha: [20.7453, 78.6022],
  Washim: [20.1135, 77.1293],
  Yavatmal: [20.3888, 78.1204]
};

const DistrictPopup = ({ district }) => {
  const [data, setData] = useState({ current: null, f1: null, f3: null, f6: null });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [hist, f1, f3, f6] = await Promise.all([
          getDistrictHistory(district),
          getForecastDistrict(district, 1),
          getForecastDistrict(district, 3),
          getForecastDistrict(district, 6)
        ]);
        
        let current = null;
        if (hist && hist.records && hist.records.length > 0) {
          current = hist.records[hist.records.length - 1].groundwater_level_m_bgl;
        }
        
        setData({ current, f1, f3, f6 });
      } catch (error) {
        console.error("Error loading district details", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [district]);

  if (loading) return <div className="p-2 w-48 text-center text-sm">Loading...</div>;

  return (
    <div className="p-1 w-56 font-sans">
      <h3 className="font-bold text-base mb-2 border-b pb-1">{district}</h3>
      <div className="text-sm space-y-1.5 mb-2">
        <div className="flex justify-between">
          <span className="text-slate-500">Current:</span>
          <span className="font-medium">{data.current ? data.current.toFixed(2) : '-'} m bgl</span>
        </div>
        {data.f1 && (
          <div className="flex justify-between">
            <span className="text-slate-500">+1 Month:</span>
            <span className="font-medium text-blue-600">{data.f1.predicted_groundwater_level_m_bgl.toFixed(2)} m bgl</span>
          </div>
        )}
        {data.f3 && (
          <div className="flex justify-between">
            <span className="text-slate-500">+3 Months:</span>
            <span className="font-medium text-blue-600">{data.f3.predicted_groundwater_level_m_bgl.toFixed(2)} m bgl</span>
          </div>
        )}
        {data.f6 && (
          <div className="flex justify-between">
            <span className="text-slate-500">+6 Months:</span>
            <span className="font-medium text-blue-600">{data.f6.predicted_groundwater_level_m_bgl.toFixed(2)} m bgl</span>
          </div>
        )}
      </div>
      <div className="text-[10px] text-slate-400 italic mt-2 border-t pt-1">
        Historical-input demonstration
      </div>
    </div>
  );
};

const MaharashtraMap = ({ districts }) => {
  return (
    <div className="h-[400px] md:h-[500px] xl:h-[550px] w-full rounded-xl overflow-hidden border border-slate-200 relative shadow-inner z-0">
      <div className="absolute top-2 right-2 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm border border-slate-200 z-[400] text-xs font-semibold text-slate-700 pointer-events-none">
        District forecast locations
      </div>
      <MapContainer center={center} zoom={6} style={{ height: '100%', width: '100%' }} zoomControl={false}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {districts.map(district => {
          const coords = districtCoords[district];
          if (!coords) return null;
          return (
            <CircleMarker 
              key={district} 
              center={coords} 
              radius={6}
              pathOptions={{ color: '#2563eb', fillColor: '#3b82f6', fillOpacity: 0.6, weight: 2 }}
            >
              <Popup>
                <DistrictPopup district={district} />
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
};

export default MaharashtraMap;
