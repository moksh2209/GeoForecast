import client from './client';

export const getModels = async () => {
  const response = await client.get('/models');
  return response.data;
};
