docker run -d -p 5000:5000 --name web_app_0 service-discovery
docker run -d -p 5001:5001 --name web_app_1 --env PORT=5001 service-discovery
docker run -d -p 5002:5002 --name web_app_2 --env PORT=5002 service-discovery