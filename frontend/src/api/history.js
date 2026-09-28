import client from './client';

export const getDistrictHistory = async (district) => {
  const response = await client.get(`/districts/${district}/history`);
  return response.data;
};
