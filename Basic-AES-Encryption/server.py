from Crypto.Cipher import AES
from secrets import token_bytes
import socket 
import pickle

PORT = 5000
SERVER = socket.gethostbyname(socket.gethostname())
FORMAT = 'utf-8'
ADDR = (SERVER, PORT)


def encrypt(msg):
    key = token_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce 
    ciphertext, tag = cipher.encrypt_and_digest(msg.encode(FORMAT))
    return key, nonce, ciphertext, tag

def decrypt(key, nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    try:
        plaintext = cipher.decrypt(ciphertext)
        cipher.verify(tag)
        return plaintext
    except:
        return False
    
def Handle_client(conn, addr, data):
    try:
        key, nonce, ciphertext, tag = pickle.loads(data)
    except pickle.UnpicklingError:
        print("[ERROR]: Invalid data format received.")
        return 
    
    try:
        plaintext = decrypt(key, nonce, ciphertext, tag)
        if plaintext:
            print(f"[MESSAGE]: {plaintext.decode(FORMAT)}")

            response = input("[SERVER]: " )
            key, nonce, ciphertext, tag = encrypt(response)
            encrypted_response = (key, nonce, ciphertext, tag)

            conn.sendall(pickle.dumps(encrypted_response))
        else:
            print("[ERROR]: Decryption failed (message corrupted or invalid)")
    except:
        print("[ERROR]: Invalid authentication tag (message tampered with)")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.bind(ADDR)

    print('[SERVER]: Server is starting...')
    print(f'[SERVER]: Server is listening on {SERVER}, on port {PORT}')
    server.listen()
    connected = True

    while connected: 
        conn, addr = server.accept()
        data = conn.recv(2048)

        print(f'[NEW CONNECTION]: {addr} connected.\n')

        Handle_client(conn,addr,data)

        data = conn.recv(2048)
        Handle_client(conn, addr, data)

        connected = False
    conn.close()
       
main() 