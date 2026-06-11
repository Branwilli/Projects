import React from "react";
import  { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline } from "@mui/material";
import Dashboard from './pages/Dashboard.js';
import Login from './pages/Login.js';
import Courses from './pages/Courses.js';
import Settings from './pages/Settings.js';
import Finance from './pages/Finance.js';
import Navbar from './component/Navbar.js';
import Sidebar from './component/Sidebar.js';
import './index.css';

const theme = createTheme({
  palette: {
    primary: {
      main: '#003366',
    },
    secondary: {
      main: '#FFCC00',
    },
  },
});

function App() {
  const [isAunthenticated, setIsAuthenticated] = React.useState(false);
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const handleLogin = () => setIsAuthenticated(true);
  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  if (!isAunthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="app">
          <Navbar toggleSidebar={toggleSidebar} />
          <Sidebar open={sidebarOpen} toggle={toggleSidebar} />
          <main className={`content ${sidebarOpen ? 'shifted' : ''}`}>
            <Routes>
              <Route path="/" element={<Dashboard />} />,
              <Route path="/courses" element={<Courses />} />,
              <Route path="/finance" element={<Finance />} />,
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
