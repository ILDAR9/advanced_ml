REDIS_PASSWORD=lolkek123
IP=95.216.168.158

redis_simple:
	docker run -d -p 6379:6379 docker.io/library/redis:latest

redis:
	docker run -d -p 6379:6379 docker.io/library/redis:latest /bin/sh -c 'redis-server --requirepass ${REDIS_PASSWORD}'

redis_insight:
	docker run -d -p 8001:8001 docker.io/redislabs/redisinsight:latest

# cat /var/run/redis.pid