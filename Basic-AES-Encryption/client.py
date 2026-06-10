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
    
def send_message(conn):
    msg = input('[CLIENT]: ')
    key, nonce, ciphertext, tag = encrypt(msg)
    encrypted_msg = (key, nonce, ciphertext, tag)

    conn.sendall(pickle.dumps(encrypted_msg))

def Handle_message(msg):
    try: 
        key, nonce, ciphertext, tag = pickle.loads(msg)
    except pickle.UnpicklingError:
        print("[ERROR]: Invalid data format received.")
        return 
        
    try:
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        try:
            plaintext = cipher.decrypt(ciphertext)
            cipher.verify(tag)
            print(f"[MESSAGE]: {plaintext.decode(FORMAT)}")

        except:
            return False
    except:
        print("[ERROR]: Invalid authentication tag (message tampered with)\n")

def start():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    connected = True
    while connected: 

        send_message(client)
        msg = client.recv(2048)

        Handle_message(msg)
        send_message(client)

        msg = client.recv(2048)
        Handle_message(msg)

        connected = False
    client.close()

start()
