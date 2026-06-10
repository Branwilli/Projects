// Modules
const express = require('express');
const http = require('http');
const path = require('path');
const nodemailer = require('nodemailer');
const socketIo = require('socket.io');

// App and Server Setup
const app = express();
const server = http.createServer(app);

// Initialize Socket.io with the server
const io = socketIo(server, {
  //Allows any website to communicate with server 
  cors: {
      origin: "*",
  }
}); 
const port = 5000;

// Middleware
app.use(express.json()); // Parse incoming request with JSON payload
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded payloads
app.use(express.static(path.join(__dirname, 'public'))); // Serve static files

// Serve the HTML page (if needed for initial load)
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'admin-email.html'));
});

// Create a Gmail transporter
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'bw6751430@gmail.com',
    pass: "vxhp tcno hvbe avly", 
  },
});

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log('A client connected.');

  // Listen for the 'send_email' event from the client
  socket.on('send_email', (emailData) => {
    // Define email options
    const mailOptions = {
      from: 'bw6751430@gmail.com',
      to: emailData.to,
      subject: emailData.subject,
      text: emailData.text,
    };

    // Send the email using nodemailer
    transporter.sendMail(mailOptions, (error, info) => {
      if (error) {
        console.error('Error sending email:', error);
        socket.emit('email_status', { status: 'error', message: 'Error sending email' });
      } else {
        console.log('Email sent:', info.response);
        socket.emit('email_status', { status: 'success', message: 'Email sent successfully!' });
      }
    });
  });

  // Handle disconnection
  socket.on('disconnect', () => {
    console.log('A client disconnected.');
  });
});

// Start the server
server.listen(port, () => {
  console.log(`Server is listening on http://localhost:${port}`);
});