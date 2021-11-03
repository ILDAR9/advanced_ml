import sys
import json

import urllib.parse
import urllib.request

outline = sys.argv[1]

if outline not in ['test', 'prod']:
    print("Not right contur")
    sys.exit(1)


url = f"http://waryak:5000/get_secret_number/{outline}"

req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
   response = response.read()

response = json.loads(response)
response = response['secret_number']

print(response)