import logging

logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format="%(asctime)s - %(message)s" 
)

def log(msg):
    print(msg)
    logging.info(msg)