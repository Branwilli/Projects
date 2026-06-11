const mysql = require('mysql2');

function generateUserId(fullName) {
    if (!fullName || typeof fullName !== "string") {
        throw new Error("A valid full name is required.");
    }

    // Split the full name into parts
    const nameParts = fullName.trim().split(" ");
    if (nameParts.length < 2) {
        throw new Error("Full name must include both first and last names.");
    }

    const firstName = nameParts[0];
    const lastName = nameParts[1];

    // Get the first letters of the first and last name
    const firstInitial = firstName.charAt(0).toUpperCase();
    const lastInitial = lastName.charAt(0).toUpperCase();

    // Generate a random 4-digit number
    const randomNumber = Math.floor(Math.random() * 10000).toString().padStart(4, "0");

    // Concatenate to form the ID
    const userId = `${firstInitial}${lastInitial}${randomNumber}`;

    return userId;
}

//Handles adding, deleting and updating client's info to the database 
class ClientService {
    constructor() {
        this.db = mysql.createPool({
            host: 'localhost',
            user: 'root',
            password: 'Sp@rks25',
            database: 'ADVProject'
        }).promise();
    }

    async getClients() {
        let query = "SELECT * FROM Clients";

        try {
            const [result] = await this.db.execute(query);
            return result;
        } catch(err) {
            console.error('Error fetching clients data: ', err);
            throw new Error('Error fetching clients data');
        } 
    }

    async getClientByName(name) {
        let query = "SELECT * FROM Clients WHERE Name=?"

        try{
            const [result] = await this.db.execute(query, [name]);
            return result[0];
        } catch(err) {
            console.error('Error fetching client data from database: ', err);
            throw new Error('Error fetching client data from database')
        }  
    }

    async getClientById(id) {
        let query = "SELECT * FROM Clients WHERE Id=?"

        try{
            const [result] = await this.db.execute(query, [id]);
            return result[0];
        } catch(err) {
            console.error('Error fetching client data from database: ', err);
            throw new Error('Error fetching client data from database')
        }  
    }

    async getClientByEmail(email){
        let query = "SELECT * FROM Clients WHERE Email=?"

        try{
            const [result] = await this.db.execute(query, [email]);
            return result[0];
        } catch(err) {
            console.error('Error fetching client data from database: ', err);
            throw new Error('Error fetching client data from database')
        }  
    }

    async addClient(name, email, password) {
        try {
            // Retrieve clients from the database
            const clients = await this.getClients();

            //Generate unique id 
            let id = generateUserId(name);

            //Check if generated ID already exist and generates new ID 
            while (clients.some(clients => clients.ID === id)) {
                id = generateUserId(name);
            }
        
            // Check if email or password already exists
            const emailExists = clients.some(client => client.Email === email);
            const passwordExists = clients.some(client => client.Password === password);
    
            if (emailExists) {
                console.log('Account already exists. Email is already in use.');
                throw new Error('Email is already in use');
            }
    
            if (passwordExists) {
                console.log('Account already exists. Password is already in use.');
                throw new Error('Password is already in use');
            }
    
            // If no conflicts, insert the new client
            const query = "INSERT INTO Clients (ID, Name, Email, Password) VALUES (?, ?, ?, ?)";
            const [result] = await this.db.execute(query, [id, name, email, password]);
            console.log('Client added successfully');
            return result;
        } catch (err) {
            console.error('Error adding client data to database: ', err.message);
            throw err;
        }
    }

    async deleteClientByName(name) {
        let query = 'DELETE FROM Clients WHERE Name=?';

        try {
            const [result] = await this.db.execute(query, [name]);
            return result;
        } catch(err) {
            console.error('Error deleting client account: ', err);
            throw new Error('Error deleting Client account');
        }
    }

    async deleteClientById(id) {
        let query = 'DELETE FROM Clients WHERE Name=?';

        try {
            const [result] = await this.db.execute(query, [id]);
            return result;
        } catch(err) {
            console.error('Error deleting client account: ', err);
            throw new Error('Error deleting Client account');
        }
    }

    async deleteClientByEmail(email) {
        let query = 'DELETE FROM Clients WHERE Name=?';

        try {
            const [result] = await this.db.execute(query, [email]);
            return result;
        } catch(err) {
            console.error('Error deleting client account: ', err);
            throw new Error('Error deleting Client account');
        }
    }
}

module.exports = ClientService;