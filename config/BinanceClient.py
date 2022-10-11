from mimetypes import init
from binance.client import Client

class BinanceClient:
	def getClient(self):
		return Client(api_key='???', api_secret='???')
	
