import React, { useState } from "react";
import { 
    Grid, 
    Paper, 
    Typography,
    IconButton,
    Badge,
    Avatar,
    Dialog,
    DialogContent,
    TextField,
    Button,
    Box
} from '@mui/material';
import {
    School as SchoolIcon,
    People as PeopleIcon,
    LibraryBooks as LibraryIcon,
    MonetizationOn as FinanceIcon,
    Message as MessageIcon,
    Send as SendIcon,
    Close as CloseIcon
} from '@mui/icons-material';

const stats =[
    { title: 'Total Students', value: '12,345', icon: <PeopleIcon fontSize="large" /> },
    { title: 'Faculty Members', value: '1,234', icon: <SchoolIcon fontSize="large" /> },
    { title: 'Courses Offered', value: '256', icon: <LibraryIcon fontSize="large" /> },
    { title: 'Annual Budget', value: '$125M', icon: <FinanceIcon fontSize="large" /> },
];

export default function Dashboard() {
    const [openChat, setOpenChat] = useState(false);
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState([{ sender: 'Support', text: 'Hello! How can we help you today?', time: '10:30 AM' }]);
    const [unread, setUnread] = useState(1);

    const handleSendMessage = async() => {
        if (message.trim()) {
            const userMessage = {
                sender: 'You',
                text: message,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              };

            setMessages([...messages, userMessage]);
            setMessage('');

            try {
                const response = await fetch('http://localhost:5000/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                if (data.response) {
                    const botMessage = {
                        sender: 'Support',
                        text: data.response,
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    };
                    setMessages(prev => [...prev, botMessage]);
                }
            } catch (error) {
                console.error('Error calling chatbot API:', error);
            }
        }
    };

    const handleOpenChat = () => {
        setOpenChat(true);
        setUnread(0);
    };

    return (
        <div>
            <Typography variant="h4" gutterBottom>
                University Dashboard
            </Typography>

            <Grid container spacing={3}>
                {stats.map((stat, index) => (
                    <Grid item xs={12} sm={6} md={3} key={index}>
                        <Paper sx={{ p: 3, display: 'flex', alignItems: 'center' }}>
                            <div style={{ marginRight: 16, color: '#003366' }}>
                                {stat.icon}
                            </div>
                            <div>
                                <Typography variant="h6">{stat.value}</Typography>
                                <Typography variant="subtitle2">{stat.title}</Typography>
                            </div>
                        </Paper>
                    </Grid>
                ))}

                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 3, height: 300 }}>
                        <Typography variant="h6" gutterBottom>
                            Enrollment Trends
                        </Typography>
                            {/* Placeholder for chart */}
                        <div style={{ 
                            backgroundColor: '#f5f5f5', 
                            height: '80%', 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center' 
                            }}>
                            <Typography>Chart Component</Typography>
                        </div>
                    </Paper>
                </Grid>

                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, height: 300 }}>
                        <Typography variant="h6" gutterBottom>
                            Recent Announcements
                        </Typography>
                        {/* Announcements list would go here */}
                    </Paper>
                </Grid>
            </Grid>

            <Box sx={{
                position: 'fixed',
                bottom: 24,
                right: 24,
                zIndex: 1000
            }}>
                <IconButton 
                    color="primary"
                    aria-label='chat'
                    onClick={handleOpenChat}
                    sx={{
                        backgroundColor: 'primary.main',
                        color: 'white',
                        width: 56,
                        height: 56,
                        boxShadow: 3,
                        '&:hover': {
                            backgroundColor: 'primary.dark',
                        }
                    }}>
                        <Badge badgeContent={unread} color="error">
                            <MessageIcon />
                        </Badge>
                </IconButton>
            </Box>

            <Dialog 
                open={openChat}
                onClose={() => setOpenChat(false)}
                PaperProps={{
                    sx: {
                        position: 'fixed',
                        bottom: 80,
                        right: 24,
                        margin: 0,
                        maxHeight: '60vh',
                        width: 350,
                        maxWidth: '90vw',
                    }
                }}
                BackdropProps={{
                    sx: {
                        backgroundColor: 'transparent'
                    }
                }}>
                    <Box sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        p: 2,
                        borderBottom: '1px solid',
                        borderColor: 'divider'
                    }}>
                        <Typography variant="h6">University Support</Typography>
                        <IconButton onClick={() => setOpenChat(false)}>
                            <CloseIcon />
                        </IconButton>
                    </Box>
                    <DialogContent sx={{ p: 2, height: 300, overflowY: 'auto' }}>
                        {messages.map((msg, index) => (
                            <Box key={index} sx={{
                                mb: 2,
                                display: 'flex',
                                flexDirection: msg.sender === 'You' ? 'row-reverse' : 'row',
                                alignItems: 'flex-start'
                            }}>
                                <Avatar sx={{ 
                                    width: 32, 
                                    height: 32,
                                    mr: msg.sender === 'You' ? 0 : 1,
                                    ml: msg.sender === 'You' ? 1 : 0,
                                    bgcolor: msg.sender === 'You' ? 'primary.main' : 'grey.500'
                                }}>
                                    {msg.sender === 'You' ? 'Y' : 'S'}
                                </Avatar>
                                <Paper sx={{ 
                                    p: 1.5,
                                    maxWidth: '70%',
                                    bgcolor: msg.sender === 'You' ? 'primary.light' : 'grey.100'
                                }}>
                                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                        {msg.sender}
                                    </Typography>
                                    <Typography variant="body1">{msg.text}</Typography>
                                    <Typography variant="caption" sx={{ 
                                        display: 'block',
                                        textAlign: 'right',
                                        color: 'text.secondary'
                                    }}>
                                        {msg.time}
                                    </Typography>
                                </Paper>
                            </Box>
                        ))}
                    </DialogContent>
                    <Box sx={{
                        p: 2, 
                        borderTop: '1px solid',
                        borderColor: 'divider',
                        display: 'flex',
                        alignItems: 'center'
                    }}>
                        <TextField
                            fullWidth
                            size="small"
                            placeholder="Type your message..."
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                        />
                        <IconButton 
                            color="primary" 
                            onClick={handleSendMessage}
                            disabled={!message.trim()}
                            sx={{ ml: 1 }}
                        >
                            <SendIcon />
                        </IconButton>
                    </Box>
                </Dialog>
        </div>
    );
}