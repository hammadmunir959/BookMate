import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Avatar,
  Menu,
  MenuItem,
  Box,
  Button,
  Chip,
} from '@mui/material';
import {
  Settings,
  Brightness4,
  Brightness7,
  Logout,
  AccountCircle,
} from '@mui/icons-material';

interface GlobalHeaderProps {
  sourceCount: number;
  selectedSourcesCount: number;
  onSettingsClick: () => void;
  onThemeToggle: () => void;
  isDarkTheme: boolean;
}

export const GlobalHeader: React.FC<GlobalHeaderProps> = ({
  sourceCount,
  selectedSourcesCount,
  onSettingsClick,
  onThemeToggle,
  isDarkTheme,
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleProfileClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileClose = () => {
    setAnchorEl(null);
  };

  return (
    <>
      <AppBar
        position="fixed"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: 'rgba(26, 26, 26, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        }}
        elevation={0}
      >
        <Toolbar>
          {/* Logo */}
          <Typography
            variant="h6"
            sx={{
              fontWeight: 800,
              fontSize: '1.75rem',
              mr: 6,
              background: 'linear-gradient(135deg, #3A6FF7 0%, #6366f1 50%, #8B5CF6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
              textShadow: '0 0 30px rgba(58, 111, 247, 0.3)',
            }}
          >
            BookMate
          </Typography>

          {/* Divider between app name and notebook content */}
          <Box sx={{ height: 32, width: '1px', bgcolor: 'rgba(255,255,255,0.2)', mr: 3 }} />

          {/* Spacer for layout */}
          <Box sx={{ flex: 1 }} />


          {/* Right side buttons - improved spacing and alignment */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, pr: 2 }}>
            {selectedSourcesCount > 0 && (
              <Chip
                label={`${selectedSourcesCount} selected`}
                size="small"
                sx={{
                  backgroundColor: 'rgba(16, 185, 129, 0.15)',
                  color: '#10B981',
                  border: '1px solid rgba(16, 185, 129, 0.4)',
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  height: '28px',
                  borderRadius: '14px',
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: 'rgba(16, 185, 129, 0.25)',
                    transform: 'translateY(-1px)',
                  },
                  transition: 'all 0.2s ease',
                }}
                onClick={() => console.log('Show selected sources modal')}
              />
            )}

            <IconButton
              onClick={onThemeToggle}
              sx={{
                color: 'rgba(255,255,255,0.8)',
                borderRadius: 2,
                p: 1,
                '&:hover': {
                  color: 'white',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                }
              }}
            >
              {isDarkTheme ? <Brightness7 /> : <Brightness4 />}
            </IconButton>

            <IconButton
              onClick={handleProfileClick}
              sx={{
                color: 'rgba(255,255,255,0.8)',
                borderRadius: 2,
                p: 1,
                '&:hover': {
                  color: 'white',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                }
              }}
            >
              <Avatar sx={{ width: 32, height: 32, bgcolor: '#3A6FF7' }}>
                <AccountCircle />
              </Avatar>
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Profile Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        PaperProps={{
          sx: {
            backgroundColor: '#2A2A2A',
            border: '1px solid #404040',
          },
        }}
      >
        <MenuItem onClick={handleProfileClose}>
          <AccountCircle sx={{ mr: 1 }} />
          Profile
        </MenuItem>
        <MenuItem onClick={() => { handleProfileClose(); onSettingsClick(); }}>
          <Settings sx={{ mr: 1 }} />
          Settings
        </MenuItem>
        <MenuItem onClick={handleProfileClose}>
          <Logout sx={{ mr: 1 }} />
          Logout
        </MenuItem>
      </Menu>
    </>
  );
};
