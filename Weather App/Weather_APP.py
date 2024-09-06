import requests
import json
import os
city = input("Enter your city : ")

url = f"http://api.weatherapi.com/v1/current.json?key=67ac44985c864486b0755003242808&q={city}"

r = requests.get(url)
# print(r.text)
w_dic = json.loads(r.text)
w = (w_dic["current"]["temp_c"])
print(w)
os.system(f"say 'The temperature of {city} is {w} degrees'")