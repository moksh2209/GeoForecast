import client from './client';

export const getDistricts = async () => {
  const response = await client.get('/districts');
  return response.data;
};

export const getDistrict = async (district) => {
  const response = await client.get(`/districts/${district}`);
  return response.data;
};
