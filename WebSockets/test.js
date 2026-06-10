const express = require('express');
const path = require('path');
const http = require('http');
const ClientService = require('./database');  // Import the ClientService class
const cors = require("cors");

const app = express();
const server = http.createServer(app);

// Initialize an instance of ClientService
const clientService = new ClientService();

// Middleware
app.use(express.json());
app.use(cors({
    origin: '*', // Allows all origins (adjust this to specify allowed origins)
    methods: ['GET', 'POST', 'PUT', 'DELETE'], // Specifies allowed HTTP methods
    //allowedHeaders: ['Content-Type', 'Authorization'], // Specifies allowed headers
}));
app.use(express.urlencoded({ extended: true }));

// Define the router
const router = express.Router();

// Serve the HTML file at the root route
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'admin-client-table.html'));
});

// Route to fetch all clients
router.get('/clients', async (req, res, next) => {
    try {
        const clients = await clientService.getClients(); // Use the ClientService instance to get all clients details
        res.send(clients);
    } catch (err) {
        next(err); // Pass errors to the error-handling middleware
    }
});

//Route to fetch a client by ID
router.get('/client/id', async (req, res, next) => {
    try {
        const data = req.query;  //stores the value of the request 

        // Check if the resquest value is empty  
        if (!data.q) {
            console.log("Input is empty");
            return res.status(400).send({ error: "Client id is required" });
        }
        const client = await clientService.getClientById(data.q);

        if (!client) {
            console.log("Client not found")
            return res.status(404).send({ error: "Client not found" });
        }
        res.send(client);
    } catch (err) {
        next(err); 
    }
});

//Route to fetch client by email
router.get('/client/email', async (req, res, next) => {
    try {
        const data = req.query;  //stores the value of the request 

        // Check if the resquest value is empty  
        if (!data.q) {
            console.log("Input is empty");
            return res.status(400).send({ error: "Client email is required" });
        }
        const client = await clientService.getClientByEmail(data.q);

        if (!client) {
            console.log("Client not found")
            return res.status(404).send({ error: "Client not found" });
        }
        res.send(client);
    } catch (err) {
        next(err); 
    }
});

//Route to fetch client by name 
router.get('/client/name', async (req, res, next) => {
    try {
        const data = req.query; 
        if (!data.q) {
            console.log("Input is empty");
            return res.status(400).send({ error: "Client name is required" });
        }
        const client = await clientService.getClientByName(data.q);

        if (!client) {
            console.log("Client not found")
            return res.status(404).send({ error: "Client not found" });
        }
        res.send(client);
        console.log(client);
    } catch (err) {
        next(err);
    }
});

// Use the router
app.use('/', router);

// Error-handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Start the server
server.listen(8080, () => {
    console.log('Server is running at http://localhost:8080');
});