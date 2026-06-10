import React from "react";
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { Dashboard as DashboardIcon,
    School as SchoolIcon,
    MonetizationOn as FinanceIcon,
    Settings as SettingsIcon 
} from '@mui/icons-material';
import { NavLink } from "react-router-dom";

const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Courses', icon: <SchoolIcon />, path: '/courses' },
    { text: 'Finance', icon: <FinanceIcon />, path: '/finance' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

export default function Sidebar({ open, toggle }) {
    return (
        <Drawer 
            variant="persistent"
            open={open}
            sx={{ 
                width: 240,
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                    width: 240,
                    boxSizing: 'border-box',
                    marginTop: '64px',
                },
            }}>
            <Divider />
            <List>
                {menuItems.map((item) => (
                    <ListItem
                        button
                        key={item.text}
                        component={NavLink}
                        to={item.path}
                        sx={{
                            '& .active': {
                                backgroundColor: 'rgba(0, 51, 102, 0.1)',
                                color: 'primary.main',
                            },
                        }}
                    >
                        <ListItemIcon>{item.icon}</ListItemIcon>
                        <ListItemText primary={item.text} />
                    </ListItem>
                ))}
            </List>
        </Drawer>
    );
}