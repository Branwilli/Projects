//Import Libraries 
const express = require('express');
const { createServer } = require('http');
const { join } = require('node:path');
const { Server } = require('socket.io');

const app = express(); //Initalize app
const server = createServer(app); //Creates http server 

//Creates WebSocket server that listen on the server
const io = new Server(server, {
  //Allows any website to communicate with server 
    cors: {
        origin: "*",
    }
});

//Creates route to server chat 
app.get('/', (req, res) => {
  res.sendFile(join(__dirname, 'chatPage.html'));
});

//Handles Socket connections and messages 
io.on('connection', (socket) => {
    console.log('a user connected', socket.id);

    socket.on('chat message', (msg) => {
      console.log('message: ' + msg);
      io.emit('chat message', msg);
    });

    socket.on('disconnect', () => {
      console.log('user disconnected');
    });
  });

//Handles Chat Messages
//io.on('connection', (socket) => {
    //socket.on('chat message', (msg) => {
      //console.log('message: ' + msg);
      //io.emit('chat message', msg);
    //});
  //});

//Starts the server
server.listen(5000, () => {
  console.log('server running at http://localhost:5000');
});