import React from 'react';
import { AppBar, Toolbar, IconButton, Typography, Badge, Avatar } from '@mui/material';
import { Menu as MenuIcon, Notifications as NotificationsIcon, AccountCircle as AccountIcon } from '@mui/icons-material';

export default function Navbar({ toggleSidebar }) {
    return (
        <AppBar position='fixed' sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
                <IconButton
                    color="inherit"
                    edge="start"
                    onClick="{toggleSidebar}"
                    sx={{ mr: 2 }}
                >
                    <MenuIcon />
                </IconButton>
                <Typography variant='h6' noWrap component="div" sx={{ flexGrow: 1 }}>
                    University Portal
                </Typography>
                <IconButton color='inherit'>
                    <Badge badgeContent={4} color='secondary'>
                        <NotificationsIcon />
                    </Badge>
                </IconButton>
                <IconButton color='inherit'>
                    <Avatar sx={{ width: 32, height: 32 }}>
                        <AccountIcon />
                    </Avatar>
                </IconButton>
            </Toolbar>
        </AppBar>
    );
}