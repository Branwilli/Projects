import React, {useState} from 'react';
import { 
    Box, 
    Typography, 
    Paper, 
    List, 
    ListItem, 
    ListItemText, 
    Divider, 
    Switch,
    TextField,
    Button,
    Avatar
} from '@mui/material';
import {
    AccountCircle as ProfileIcon,
    Notifications as NotificationsIcon,
    Security as SecurityIcon,
    Palette as ThemeIcon
} from '@mui/icons-material';

export default function Settings() {
    const [notifications, setNotifications] = useState(true);
    const [darkMode, setDarkMode] = useState(false);
    const [activeTab, setActiveTab] = useState('profile');

    const handleTabChange = (tab) => {
        setActiveTab(tab);
    };

    return (
        <Box sx={{ display: 'flex', p: 3, gap: 3 }}>
            <Paper sx={{ width: 250 }}>
                <List>
                    <ListItem
                        button 
                        selected={activeTab === 'profile'}
                        onClick={() => handleTabChange('profile')}
                        >
                            <ProfileIcon sx={{ mr: 2 }} />
                            <ListItemText primary="Profile" />
                    </ListItem>
                    <ListItem 
                        button 
                        selected={activeTab === 'notifications'}
                        onClick={() => handleTabChange('notifications')}
                    >
                        <NotificationsIcon sx={{ mr: 2 }} />
                        <ListItemText primary="Notifications" />
                    </ListItem>
                    <ListItem 
                        button 
                        selected={activeTab === 'security'}
                        onClick={() => handleTabChange('security')}
                    >
                        <SecurityIcon sx={{ mr: 2 }} />
                        <ListItemText primary="Security" />
                    </ListItem>
                    <ListItem 
                        button 
                        selected={activeTab === 'theme'}
                        onClick={() => handleTabChange('theme')}
                    >
                        <ThemeIcon sx={{ mr: 2 }} />
                        <ListItemText primary="Theme" />
                    </ListItem>
                </List>
            </Paper>

            <Box sx={{ flexGrow: 1 }}>
                <Typography variant='h4' gutterBottom>
                    Settings
                </Typography>
                <Paper sx={{ p: 3 }}>
                    {activeTab === 'profile' && (
                        <Box>
                            <Typography variant='h6' gutterBottom>
                                Profile Information
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                                <Avatar sx={{ width: 80, height: 80, mr: 3 }} />
                                <Button variant='outlined'>Change Photo</Button>
                            </Box>
                            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3 }}>
                                <TextField label="First Name" fullWidth defaultValue="John" />
                                <TextField label="Last Name" fullWidth defaultValue="Doe" />
                                <TextField label="Email" fullWidth defaultValue="john.doe@university.edu" />
                                <TextField label="Phone" fullWidth defaultValue="+1 (555) 123-4567" />
                            </Box>
                            <Button variant="contained" sx={{ mt: 3 }}>Save Changes</Button>
                        </Box>
                    )}

                    {activeTab === 'notifications' && (
                        <Box>
                            <Typography variant="h6" gutterBottom>
                                 Notification Preferences
                            </Typography>
                            <List>
                                <ListItem>
                                    <ListItemText
                                        primary='Email Notifications'
                                        secondary='Receive important updates via email'
                                    />
                                    <Switch 
                                        checked={notifications} 
                                        onChange={() => setNotifications(!notifications)} 
                                    />
                                </ListItem>
                                <Divider />
                                <ListItem>
                                    <ListItemText 
                                        primary="System Notifications" 
                                        secondary="Show notifications within the system" 
                                    />
                                    <Switch 
                                        checked={notifications} 
                                        onChange={() => setNotifications(!notifications)} 
                                    />
                                </ListItem>
                            </List>
                        </Box>
                    )}

                    {activeTab === 'theme' && (
                        <Box>
                            <Typography variant='h6' gutterBottom>
                                Theme Preferences
                            </Typography>
                            <List>
                                <ListItem>
                                    <ListItemText
                                        primary='Dark Mode'
                                        secondary='Switch between light and dark theme'
                                    />
                                    <Switch
                                        checked={darkMode}
                                        onChange={() => setDarkMode(!darkMode)}
                                    />
                                </ListItem>
                            </List>
                        </Box>
                    )}

                    {activeTab === 'security' && (
                        <Box>
                            <Typography variant='h6' gutterBottom>
                                Security Settings
                            </Typography>
                            <TextField
                                label="Current Password"
                                type="password"
                                fullWidth
                                sx={{ mb: 2 }}
                            />
                            <TextField
                                label="New Password"
                                type="password"
                                fullWidth
                                sx={{ mb: 2 }}
                            />
                            <TextField
                                label="Confirm New Password"
                                type="password"
                                fullWidth
                                sx={{ mb: 3 }}
                            />
                            <Button variant="contained">Change Password</Button>
                        </Box>
                    )}
                </Paper>
            </Box>
        </Box>
    );
}