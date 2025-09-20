import React from 'react';
import { Box, Typography } from '@mui/material';

export const RightPanel: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6">Right Panel</Typography>
      <Typography variant="body2" color="text.secondary">
        This component is under development.
      </Typography>
    </Box>
  );
};
