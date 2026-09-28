import client from './client';

export const getForecastMaharashtra = async (horizon) => {
  const response = await client.get(`/forecast/maharashtra?horizon=${horizon}`);
  return response.data;
};

export const getForecastDistrict = async (district, horizon) => {
  const response = await client.get(`/forecast/${district}?horizon=${horizon}`);
  return response.data;
};
