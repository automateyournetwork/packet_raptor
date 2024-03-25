# packet_raptor
An AI Assistant using RAPTOR to talk to .pcaps

## Getting started

Clone the repo

Make a .env file inside the packet_raptor folder (where the packet_raptor.py file is located) or add the key to your docker-compose file environment settings

put this in the file:
```console
OPENAI_API_KEY="<your openapi api key>"
```

## Bring up the server
docker-compose up 

## Visit localhost
http://localhost:8585

### Usage
This has been tested with a variety of larger .pcap files and works best with larger data sets. For smaller PCAPS Packet Buddy should work fine
